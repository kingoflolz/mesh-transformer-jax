import gc
import random
import time
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.maps import thread_resources
from jax.experimental.pjit import pjit

from mesh_transformer.checkpoint import read_ckpt, write_ckpt, write_ckpt_v2, load_ckpt_v2
from mesh_transformer.layers import EmbeddingShard, TransformerLayerShard, RelativePositionEmbs, ProjectionShard, \
    TransformerLayerShardV2, Projection, EmbeddingShardV2
from mesh_transformer.util import to_f32, to_bf16, maybe_shard, head_print, global_norm
from jax.experimental import PartitionSpec as P


class CausalTransformerShard(hk.Module):
    def __init__(self, config):
        super().__init__()
        heads = config["n_heads"]
        shards = config["cores_per_replica"]
        layer_count = config["layers"]

        self.transformer_layers = []
        self.heads = heads

        self.heads_per_shard = heads // shards

        self.embed = EmbeddingShard(config)

        init_scale = 2. / layer_count

        for i in range(layer_count):
            self.transformer_layers.append(TransformerLayerShard(config, name=f"layer_{i}", init_scale=init_scale))

        self.proj = ProjectionShard(config)

        if config["pe"] == "t5":
            self.rpe = RelativePositionEmbs()
        else:
            self.rpe = None

    def eval(self, context, target, z_loss=0., mask=0.0):
        input_len = context.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
        else:
            attn_bias = 0

        attn_bias += mask

        x = hk.remat(self.embed)(context)

        for l in self.transformer_layers:
            x = x + hk.remat(l)(x, attn_bias)

        return hk.remat(self.proj.loss)(x, target, z_loss)

    def loss(self, ctx, tgt, z_loss=False, mask=0.0):
        loss, correct = self.eval(ctx, tgt, float(z_loss), mask=mask)

        return {
            "loss": loss.mean(),
            "last_loss": loss[-1].mean(),
            "all_loss": loss,
            "correct": correct
        }

    def generate_initial(self, context, length):
        # slice last token off the context (we use that in generate_once to generate the first new token)
        last = context[-1:]
        context = context[:-1]

        input_len = context.shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
        else:
            attn_bias = 0

        x = self.embed(context)

        states = []

        for l in self.transformer_layers:
            res, layer_state = l.get_init_decode_state(x, length - 1, attn_bias)
            x = x + res
            states.append(layer_state)

        return self.proj(x), (last.astype(jnp.uint32), states, hk.next_rng_key())

    def generate_once(self, new_tok, state):
        input_len = state[0]["v"].shape[0]

        if self.rpe is not None:
            attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
            attn_bias = attn_bias[:, -1:, :]
        else:
            attn_bias = 0

        x = self.embed(new_tok)

        new_states = []

        for l, s in zip(self.transformer_layers, state):
            res, layer_state = l.decode_once(s, x, attn_bias)
            x = x + res
            new_states.append(layer_state)

        return self.proj(x), new_states


class CausalTransformer:
    def __init__(self, config):
        self.config = config
        optimizer = config["optimizer"]

        def eval(state, ctx, tgt, ctx_length):
            def eval_loss(x, y, mask):
                transformer = CausalTransformerShard(config)
                return transformer.loss(x, y, mask=mask)

            eval_loss_fn = hk.without_apply_rng(hk.transform(eval_loss)).apply

            mask = (jnp.arange(0, len(ctx)) > ctx_length) * -1e10

            return eval_loss_fn(to_bf16(state["params"]), ctx, tgt, mask)

        def train(state, ctx, tgt):
            def train_loss(x, y):
                transformer = CausalTransformerShard(config)
                out = transformer.loss(x, y, z_loss=True)

                return out["loss"], out["last_loss"]

            train_loss_fn = hk.without_apply_rng(hk.transform(train_loss)).apply

            def microbatch(old_grad, batch):
                ctx, tgt = batch

                val_grad_fn = jax.value_and_grad(train_loss_fn, has_aux=True)
                (loss, last_loss), grad = val_grad_fn(to_bf16(state["params"]), ctx, tgt)

                new_grad = jax.tree_multimap(lambda a, b: a + b, old_grad, grad)
                gnorm = global_norm(grad)
                return new_grad, (loss, last_loss, gnorm)

            if ctx.shape[0] == 1:
                val_grad_fn = jax.value_and_grad(train_loss_fn, has_aux=True)
                (loss, last_loss), grad = val_grad_fn(to_bf16(state["params"]), ctx[0], tgt[0])
                gnorm = global_norm(grad)
            else:
                grad, (loss, last_loss, gnorm) = jax.lax.scan(microbatch,
                                                       jax.tree_map(lambda x: jnp.zeros_like(x).astype(jnp.bfloat16),
                                                                    state["params"]),
                                                       (ctx, tgt))

            grad_norm_micro = jax.lax.pmean(gnorm, "batch")

            grad = jax.lax.pmean(grad, "batch")
            grad_norm = global_norm(grad)
            updates, new_opt_state = optimizer.update(grad, state["opt_state"], state["params"])

            return to_f32(loss), to_f32(last_loss), to_f32(grad_norm), to_f32(grad_norm_micro), {
                "params": optax.apply_updates(state["params"], to_f32(updates)),
                "step": state["step"] + 1,
                "opt_state": new_opt_state
            }

        def init(key, x):
            def train_loss(x, y):
                transformer = CausalTransformerShard(config)
                return transformer.loss(x, y)

            param_init_fn = hk.transform(hk.experimental.optimize_rng_use(train_loss)).init

            params = param_init_fn(key, x, x)

            return {
                "params": ("early_cast" in config and to_bf16 or to_f32)(params),
                "step": np.array(0),
                "opt_state": optimizer.init(params)
            }

        def generate(state, key, ctx, ctx_length, aux, sampler_options):
            sampler = config["sampler"]
            gen_length = self.gen_length

            def generate_sample(context, ctx_length, aux):
                transformer = CausalTransformerShard(config)
                _, initial_state = transformer.generate_initial(context, ctx_length)

                def generate_scan_fn(carry, sampler_input):
                    next_token, decode_state, sample_key = carry
                    sample_key, new_key = jax.random.split(sample_key)

                    logits, new_state = transformer.generate_once(next_token, decode_state)
                    next_token, sample_info = sampler(sample_key, logits, sampler_input, **sampler_options)

                    if self.return_logits:
                        output = (next_token, sample_info, logits)
                    else:
                        output = (next_token, sample_info)
                    new_carry = (next_token, new_state, new_key)
                    return new_carry, output

                final_state, outputs = jax.lax.scan(generate_scan_fn, initial_state, xs=aux, length=gen_length)
                return final_state, outputs

            generate_fn = hk.transform(generate_sample).apply
            return generate_fn(state["params"], key, ctx, ctx_length, aux)

        self.init_xmap = jax.experimental.maps.xmap(fun=init,
                                                    in_axes=(["shard", ...],
                                                             ["batch", ...]),
                                                    out_axes=["shard", ...],
                                                    axis_resources={'shard': 'mp', 'batch': 'dp'})

        self.eval_xmap = jax.experimental.maps.xmap(fun=eval,
                                                    in_axes=(["shard", ...],
                                                             ["batch", ...],
                                                             ["batch", ...],
                                                             ["batch", ...]),
                                                    out_axes=["batch", ...],
                                                    axis_resources={'shard': 'mp', 'batch': 'dp'})

        self.train_xmap = jax.experimental.maps.xmap(fun=train,
                                                     in_axes=(["shard", ...],
                                                              ["batch", ...],
                                                              ["batch", ...]),
                                                     out_axes=(["batch", ...], ["batch", ...], ["batch", ...], ["batch", ...], ["shard", ...]),
                                                     donate_argnums=(0,),
                                                     axis_resources={'shard': 'mp', 'batch': 'dp'})

        self.generate_xmap = jax.experimental.maps.xmap(fun=generate,
                                                        in_axes=(["shard", ...],
                                                                 ["batch", ...],
                                                                 ["batch", ...],
                                                                 ["batch", ...],
                                                                 ["batch", ...],
                                                                 ["batch", ...]),
                                                        out_axes=["batch", ...],
                                                        axis_resources={'shard': 'mp', 'batch': 'dp'})

        self.move_xmap = jax.experimental.maps.xmap(fun=lambda x, _: to_bf16(x),
                                                    in_axes=(["shard", ...], ["batch", ...]),
                                                    out_axes=["shard", ...],
                                                    axis_resources={'shard': 'mp', 'batch': 'dp'})

        key = hk.PRNGSequence(42)

        assert thread_resources.env.shape['mp'] == config["cores_per_replica"]

        dp = thread_resources.env.shape['dp']
        mp = thread_resources.env.shape['mp']

        mp_per_host = min(mp, 8)

        seq = config["seq"]
        vocab = config["n_vocab"]

        example_shape = (max(dp // jax.host_count(), 1), seq,)
        x = jax.random.uniform(next(key), example_shape, minval=0, maxval=vocab).astype(jnp.uint32)  # batch, len

        head_print("key shape", jnp.array(key.take(mp_per_host)).shape)
        head_print("in shape", x.shape)

        head_print("dp", dp)
        head_print("mp", mp)

        self.gen_length = 1
        self.state = self.init_xmap(jnp.array(key.take(mp_per_host)), x)

        param_count = hk.data_structures.tree_size(self.state['params'])
        head_print(f"Total parameters: {param_count}")

    def write_ckpt(self, path, shard):
        write_ckpt(self.state, path, shard)

    def load_ckpt(self, path):
        self.state = read_ckpt(self.state, path, thread_resources.env.shape['mp'])

    def train(self, sample):
        # print("train iter")
        # print("sample", sample["obs"])
        # print("target", sample["target"])
        obs = jnp.transpose(sample["obs"], (1, 0, 2))
        target = jnp.transpose(sample["target"], (1, 0, 2))

        # print("train sample", obs.shape)
        # print("train target", target.shape)

        # assert (sample["obs"][:, 1:] == sample["target"][:, -1])

        # start = time.time()
        loss, last_loss, grad_norm, grad_norm_micro, self.state = self.train_xmap(self.state, obs, target)
        loss = np.array(loss)
        last_loss = np.array(last_loss)
        grad_norm = np.array(grad_norm)
        # print(f"iter done in {time.time() - start:.06}s")
        return loss.mean(), last_loss.mean(), grad_norm.mean(), grad_norm_micro.mean()

    def eval(self, sample):
        # print("eval sample", sample["obs"].shape)
        # print("eval target", sample["target"].shape)

        # start = time.time()

        if "ctx_length" in sample:
            ctx_length = sample["ctx_length"]
        else:
            ctx_length = np.array([len(sample["obs"][0])] * len(sample["obs"]))

        out = self.eval_xmap(self.state, sample["obs"], sample["target"], ctx_length)
        # print(f"eval dispatched in {time.time() - start:.06}s")

        # np.array(out["loss"])
        # print(f"eval done in {time.time() - start:.06}s")
        return out

    def generate(self, ctx, ctx_length, gen_length, sampler_options, return_logits=False):
        key = hk.PRNGSequence(random.randint(0, 2 ** 60))

        batch_size = ctx.shape[0]
        aux = jnp.zeros((batch_size, gen_length), dtype=jnp.uint32)
        self.gen_length = gen_length
        self.return_logits = return_logits

        return self.generate_xmap(self.state,
                                  jnp.array(key.take(batch_size)),
                                  ctx,
                                  np.array(ctx_length, dtype=np.uint32),
                                  aux,
                                  sampler_options)


# this bypasses the CausalTransformerShard class (which causes ugly code) but in return allows layers to be processed
# by a `jax.scan`, which allows for much faster and O(1) compile times w.r.t. layers.
class CausalTransformerV2:
    def __init__(self, config):
        self.config = config
        optimizer = config["optimizer"]
        with_optimizer = optimizer is not None

        head_print(f"with_optimizer: {with_optimizer}")

        bf16_optimizer = config.get("bf16_optimizer", False)
        early_cast = config.get("early_cast", False)
        early_collect = config.get("early_collect", True)

        def embedding(x):
            x = maybe_shard(x, P("dp", None))
            return EmbeddingShardV2(config)(x)

        def residual(x, mask):
            out = x + TransformerLayerShardV2(config, init_scale=2. / config["layers"])(x, mask)
            return maybe_shard(out, P("dp", None, "mp"))

        def init_decode(x, given_length, mask):
            residual, decode_state = TransformerLayerShardV2(config, init_scale=2. / config["layers"])\
                .get_init_decode_state(x, given_length, mask)
            out = x + residual
            return maybe_shard(out, P("dp", None, "mp")), decode_state

        def iter_decode(decode_state, x):
            residual, decode_state = TransformerLayerShardV2(config, init_scale=2. / config["layers"])\
                .decode_once(decode_state, x, 0)
            out = x + residual
            return maybe_shard(out, P("dp", None, "mp")), decode_state

        def transformer(x, mask):
            return hk.remat(residual, prevent_cse=False)(x, mask)

        def projection(x):
            return Projection(config)(x)

        def init_fns():
            embed_init_fn = hk.transform(hk.experimental.optimize_rng_use(embedding)).init
            transformer_init_fn = hk.transform(hk.experimental.optimize_rng_use(transformer)).init
            projection_init_fn = hk.transform(hk.experimental.optimize_rng_use(projection)).init

            return embed_init_fn, transformer_init_fn, projection_init_fn

        def shard_strategy(shape_dtype, parallel):
            if shape_dtype.ndim == 0:
                return P()
            if shape_dtype.ndim == 1:
                return P(None)
            # embedding/projection layers
            elif shape_dtype.shape == (config["n_vocab"], config["d_model"]):
                return P(parallel, None)
            elif shape_dtype.shape == (config["d_model"], config["n_vocab"]):
                return P(None, parallel)

            # a transformer layer
            elif shape_dtype.shape[0] == config["layers"]:
                if shape_dtype.ndim == 2:
                    # a channel wise variable (e.g. layernorm parameters)
                    # replicate it for speed
                    return P(None, None)
                elif shape_dtype.ndim == 3:
                    # a weight matrix
                    matrix_size = shape_dtype.shape[1:]

                    assert matrix_size[0] != matrix_size[1]  # this case is ambiguous

                    if matrix_size[0] == config["d_model"]:
                        # shard along the axis which is _not_ the model dimension
                        return P(None, None, parallel)
                    elif matrix_size[1] == config["d_model"]:
                        return P(None, parallel, None)
                else:
                    raise NotImplementedError("borked")

            else:
                raise NotImplementedError("borked")

        def init(key, x):
            embed_init_fn, transformer_init_fn, projection_init_fn = init_fns()

            def init_scan_fn(key, x):
                new_key, key = jax.random.split(key)

                return new_key, transformer_init_fn(key, x, 0)

            e_key, t_key, p_key = jax.random.split(key, 3)

            input_shape = (config["layers"],) + x.shape + (config["d_model"],)

            params = {
                "embed": embed_init_fn(e_key, x),
                "transformer": jax.lax.scan(init_scan_fn,
                                            t_key,
                                            xs=jax.random.uniform(t_key, input_shape, dtype=jnp.float32))[1],
                "proj": projection_init_fn(p_key, jax.random.uniform(t_key, input_shape[1:], dtype=jnp.float32)),
            }

            output_state = {
                "params": (to_bf16 if early_cast else to_f32)(params),
                "step": np.array(0),
            }

            if with_optimizer:
                output_state["opt_state"] = optimizer.init((to_bf16 if bf16_optimizer else to_f32)(params))

            return output_state

        assert thread_resources.env.shape['mp'] == config["cores_per_replica"]

        dp = thread_resources.env.shape['dp']
        mp = thread_resources.env.shape['mp']

        key = hk.PRNGSequence(42)
        x = jax.random.uniform(next(key), (mp * dp, 16), minval=0, maxval=1).astype(jnp.uint32)  # batch, seq

        head_print("starting shape evaluation")

        param_shapes = jax.eval_shape(init, jax.random.PRNGKey(42), x)

        state_shard = {
                "step": P(),

                # fp32 params are also sharded (so this is like a weird mix between zero-1 and zero-3...)
                "params": jax.tree_map(partial(shard_strategy, parallel=["mp", "dp"]), param_shapes["params"]),
            }

        if "opt_state" in param_shapes:
            # zero level 1: shard optimizer states over both MP and DP
            state_shard["opt_state"] = jax.tree_map(partial(shard_strategy, parallel=["mp", "dp"]), param_shapes["opt_state"])

        self.state_shard = state_shard

        head_print("sharding strategy:")
        # head_print("state shard: ", state_shard)
        # head_print("param_shapes: ", param_shapes)
        jax.tree_multimap(head_print, state_shard, param_shapes)

        self.init_pjit = pjit(init, in_axis_resources=(None, P("dp")), out_axis_resources=state_shard)

        def apply_fns():
            embed_apply_fn = hk.without_apply_rng(hk.transform(embedding)).apply
            transformer_apply_fn = hk.without_apply_rng(hk.transform(transformer)).apply

            return embed_apply_fn, transformer_apply_fn

        def train_apply_fn(params, x, y):
            embed_apply_fn, transformer_apply_fn = apply_fns()

            def train_loss(x, y):
                loss, _ = Projection(config).loss(x, y, z_loss=1.0)
                return loss.mean(), loss[:, -1].mean()

            projection_apply_fn = hk.without_apply_rng(hk.transform(train_loss)).apply

            x = embed_apply_fn(params["embed"], x)
            x = to_bf16(x)

            def apply_scan_fn(x, layer_state):
                return to_bf16(transformer_apply_fn(layer_state, x, 0)), None

            x = jax.lax.scan(apply_scan_fn,
                             x,
                             xs=params["transformer"])[0]

            return projection_apply_fn(params["proj"], x, y)

        mp_shard_strategy = jax.tree_map(partial(shard_strategy, parallel=["mp"]), param_shapes["params"])

        def train(state, ctx, tgt):
            if early_collect:
                bf16_params = maybe_shard(to_bf16(state["params"]), mp_shard_strategy)
            else:
                bf16_params = to_bf16(state["params"])

            def microbatch(old_grad, batch):
                ctx, tgt = batch

                val_grad_fn = jax.value_and_grad(train_apply_fn, has_aux=True, allow_int=True)
                (loss, last_loss), grad = val_grad_fn(bf16_params, ctx, tgt)

                new_grad = jax.tree_multimap(lambda a, b: a + b, old_grad, grad)
                return new_grad, (loss, last_loss)

            if ctx.shape[0] == 1:
                val_grad_fn = jax.value_and_grad(train_apply_fn, has_aux=True, allow_int=True)
                (loss, last_loss), grad = val_grad_fn(bf16_params, ctx[0], tgt[0])
            else:
                grad, (loss, last_loss) = jax.lax.scan(microbatch,
                                                       jax.tree_map(lambda x: jnp.zeros_like(x).astype(jnp.bfloat16),
                                                                    bf16_params),
                                                       (ctx, tgt))

            updates, new_opt_state = optimizer.update(grad, state["opt_state"], state["params"])

            return to_f32(loss), to_f32(last_loss), {
                "params": optax.apply_updates(state["params"], to_f32(updates)),
                "step": state["step"] + 1,
                "opt_state": new_opt_state,
            }

        self.train_pjit = pjit(train,
                               in_axis_resources=(state_shard, P(None, "dp"), P(None, "dp")),
                               out_axis_resources=(None, None, state_shard),
                               donate_argnums=(0,))

        def eval_apply_fn(params, x, y, mask):
            embed_apply_fn, transformer_apply_fn = apply_fns()

            if early_collect:
                bf16_params = maybe_shard(to_bf16(params), mp_shard_strategy)
            else:
                bf16_params = to_bf16(params)

            def eval_loss(x, y):
                loss, correct = Projection(config).loss(x, y)
                return {
                            "loss": loss.mean(axis=-1),
                            "last_loss": loss[:, -1],
                            "all_loss": loss,
                            "correct": correct
                        }

            projection_apply_fn = hk.without_apply_rng(hk.transform(eval_loss)).apply

            x = embed_apply_fn(bf16_params["embed"], x)

            def apply_scan_fn(layer_in, layer_state):
                x, mask = layer_in
                return (to_bf16(transformer_apply_fn(layer_state, x, mask)), mask), None

            x = jax.lax.scan(apply_scan_fn,
                             (to_bf16(x), mask),
                             xs=bf16_params["transformer"])[0][0]

            return projection_apply_fn(bf16_params["proj"], x, y)

        def eval(params, ctx, tgt, ctx_length):
            mask = (jnp.arange(0, ctx.shape[1])[None, :] > ctx_length[:, None]) * -1e10

            # head_print("mask.shape", mask.shape)
            # head_print("ctx.shape", ctx.shape)
            # head_print("ctx_length.shape", ctx_length.shape)

            return eval_apply_fn(params, ctx, tgt, mask[:, None, None, :])

        self.eval_pjit = pjit(eval,
                              in_axis_resources=(mp_shard_strategy if early_collect else state_shard["params"],
                                                 P("dp"), P("dp"), P("dp")),
                              out_axis_resources=P("dp"))

        def generate(params, key, ctx, ctx_length, aux, sampler_options):
            sampler = config["sampler"]
            gen_length = config["gen_length"]

            embed_apply_fn, _ = apply_fns()
            init_decode_apply = hk.without_apply_rng(hk.transform(init_decode)).apply
            iter_decode_apply = hk.without_apply_rng(hk.transform(iter_decode)).apply

            def get_inital(params, ctx, ctx_length):
                x = embed_apply_fn(params["embed"], ctx)
                mask = (jnp.arange(0, ctx.shape[1])[None, :] > ctx_length[:, None]) * -1e10

                def apply_scan_fn(layer_in, layer_state):
                    x, mask = layer_in

                    x, decode_state = init_decode_apply(layer_state, x, mask)
                    return (x, mask), decode_state

                _, init_state = jax.lax.scan(apply_scan_fn,
                                 (to_bf16(x), mask),
                                 xs=params["transformer"])

                return (last.astype(jnp.uint32), init_state, hk.next_rng_key())

            initial_state = get_inital(params, ctx, ctx_length)
            initial_carry = ()

            def generate_scan_fn(carry, sampler_input):
                next_token, decode_state, sample_key = carry
                sample_key, new_key = jax.random.split(sample_key)

                x = embed_apply_fn(params["embed"], next_token)
                mask = (jnp.arange(0, ctx.shape[1])[None, :] > ctx_length[:, None]) * -1e10

                def layer_scan_fn(carry_in, layer_in):
                    x, mask = carry_in
                    layer_state, decode_state = layer_in

                    x, decode_state = iter_decode_apply(layer_state, decode_state, x)
                    return (x, mask), decode_state

                (x, _), new_state = jax.lax.scan(layer_scan_fn,
                                             (to_bf16(x), mask),
                                             xs=params["transformer"])

                projection_apply_fn = hk.without_apply_rng(hk.transform(Projection(config))).apply

                logits = projection_apply_fn(params["proj"], x)
                next_token, sample_info = sampler(sample_key, logits, sampler_input, **sampler_options)

                new_carry = (next_token, new_state, new_key)
                return new_carry, (next_token, sample_info)

            final_state, outputs = jax.lax.scan(generate_scan_fn, initial_state, xs=aux, length=gen_length)
            return final_state, outputs

        self.move_weights_pjit = pjit(lambda x: to_bf16(x),
                                      in_axis_resources=(state_shard["params"], ),
                                      out_axis_resources=mp_shard_strategy if early_collect else state_shard["params"])

        seq = config["seq"]
        vocab = config["n_vocab"]

        example_shape = (max(dp // jax.host_count(), 1), seq,)
        x = jax.random.uniform(next(key), example_shape, minval=0, maxval=vocab).astype(jnp.uint32)  # batch, len

        head_print("in shape", x.shape)

        head_print("dp", dp)
        head_print("mp", mp)

        self.state = self.init_pjit(next(key), x)
        self.state_shard = state_shard

        if with_optimizer:
            self.eval_weights = None
        else:
            self.eval_weights = self.state["params"]

        param_count = hk.data_structures.tree_size(self.state['params'])
        head_print(f"Total parameters: {param_count * dp}")

    def write_ckpt(self, path, _):
        write_ckpt_v2(self.state, path)

    def load_ckpt(self, path):
        self.state = load_ckpt_v2(self.state, path, self.state_shard, not self.config.get("eval_only", False))

    def train(self, sample):
        # print("train iter")
        # print("sample", sample["obs"])
        # print("target", sample["target"])

        obs = sample["obs"]
        target = sample["target"]

        if self.eval_weights is not None:
            self.eval_weights = None
            gc.collect()
            head_print("deleted eval weights")

        # print("train sample", obs.shape)
        # print("train target", target.shape)

        # assert (sample["obs"][:, 1:] == sample["target"][:, -1])

        start = time.time()
        loss, last_loss, self.state = self.train_pjit(self.state, obs, target)
        loss = np.array(loss)
        last_loss = np.array(last_loss)

        # head_print(f"iter done in {time.time() - start:.06}s")
        return loss.mean(), last_loss.mean()

    def eval(self, sample):
        # head_print("eval sample", sample["obs"].shape)
        # print("eval target", sample["target"].shape)

        start = time.time()

        if self.eval_weights is None:
            self.eval_weights = self.move_weights_pjit(self.state["params"])

            # blocking
            jnp.zeros(()).block_until_ready()

            head_print(f"created eval weights in {time.time() - start:.06}s")

        if "ctx_length" in sample:
            ctx_length = sample["ctx_length"]
        else:
            ctx_length = np.array([len(sample["obs"][0])] * len(sample["obs"]))

        # head_print("ctx_length in eval", ctx_length)

        out = self.eval_pjit(self.eval_weights, sample["obs"], sample["target"], ctx_length)
        # print(f"eval dispatched in {time.time() - start:.06}s")

        # np.array(out["loss"])
        # print(f"eval done in {time.time() - start:.06}s")
        return out
