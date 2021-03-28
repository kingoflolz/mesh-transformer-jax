import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.maps import thread_resources

from mesh_transformer.layers import EmbeddingShard, TransformerLayerShard, RelativePositionEmbs, ProjectionShard
from mesh_transformer.util import to_f32, to_bf16


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
        self.rpe = RelativePositionEmbs()

    def eval(self, context, target, z_loss=0.):
        input_len = context.shape[0]

        attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)

        x = hk.remat(self.embed)(context)

        for l in self.transformer_layers:
            x = x + hk.remat(l)(x, attn_bias)

        return hk.remat(self.proj.loss)(x, target, z_loss)

    def loss(self, ctx, tgt, z_loss=False):
        loss = self.eval(ctx, tgt, float(z_loss))

        return loss.mean(), loss[-1].mean()  # return lass token loss as well to check for cheating


class CausalTransformer:
    def __init__(self, config):
        self.config = config
        optimizer = config["optimizer"]

        def eval(state, ctx, tgt):
            def eval_loss(x, y):
                transformer = CausalTransformerShard(config)
                return transformer.loss(x, y)

            eval_loss_fn = hk.without_apply_rng(hk.transform(eval_loss)).apply

            return eval_loss_fn(to_bf16(state["params"]), ctx, tgt)

        def train(state, ctx, tgt):
            def train_loss(x, y):
                transformer = CausalTransformerShard(config)
                return transformer.loss(x, y, z_loss=True)

            train_loss_fn = hk.without_apply_rng(hk.transform(train_loss)).apply

            def microbatch(old_grad, batch):
                ctx, tgt = batch
                (loss, last_loss), grad = jax.value_and_grad(train_loss_fn, has_aux=True)(to_bf16(state["params"]), ctx, tgt)

                new_grad = jax.tree_multimap(lambda a, b: a + b, old_grad, grad)
                return new_grad, (loss, last_loss)

            grad, (loss, last_loss) = jax.lax.scan(microbatch,
                                        jax.tree_map(lambda x: jnp.zeros_like(x).astype(jnp.bfloat16),
                                                     state["params"]),
                                        (ctx, tgt))

            grad = jax.lax.pmean(grad, "batch")
            updates, new_opt_state = optimizer.update(grad, state["opt_state"], state["params"])

            return to_f32(loss), to_f32(last_loss), {
                "params": optax.apply_updates(state["params"], to_f32(updates)),
                "step": state["step"] + 1,
                "opt_state": new_opt_state,
            }

        def init(key, x):
            def train_loss(x, y):
                transformer = CausalTransformerShard(config)
                return transformer.loss(x, y)

            param_init_fn = hk.transform(train_loss).init

            params = param_init_fn(key, x, x)

            return {
                "params": to_f32(params),
                "step": np.array(0),
                "opt_state": optimizer.init(params)
            }

        self.init_xmap = jax.experimental.maps.xmap(fun=init,
                                                    in_axes=(["shard", ...],
                                                             ["batch", ...]),
                                                    out_axes=["shard", ...],
                                                    axis_resources={'shard': 'mp', 'batch': 'dp'})

        self.eval_xmap = jax.experimental.maps.xmap(fun=eval,
                                                    in_axes=(["shard", ...],
                                                             ["batch", ...],
                                                             ["batch", ...]),
                                                    out_axes=["batch", ...],
                                                    axis_resources={'shard': 'mp', 'batch': 'dp'})

        self.train_xmap = jax.experimental.maps.xmap(fun=train,
                                                     in_axes=(["shard", ...],
                                                              ["batch", ...],
                                                              ["batch", ...]),
                                                     out_axes=(["batch", ...], ["batch", ...], ["shard", ...]),
                                                     donate_argnums=(0,),
                                                     axis_resources={'shard': 'mp', 'batch': 'dp'})

        key = hk.PRNGSequence(42)

        assert thread_resources.env.shape['mp'] == config["cores_per_replica"]

        dp = thread_resources.env.shape['dp']
        mp = thread_resources.env.shape['mp']
        seq = config["seq"]
        vocab = config["n_vocab"]

        example_shape = (dp // jax.host_count(), seq,)
        x = jax.random.uniform(next(key), example_shape, minval=0, maxval=vocab).astype(jnp.int32)  # batch, len

        print("key shape", jnp.array(key.take(mp)).shape)
        print("in shape", x.shape)

        print("dp", dp)
        print("mp", mp)

        self.state = self.init_xmap(jnp.array(key.take(mp)), x)

    def train(self, sample):
        # print("train iter")
        # print("sample", sample["obs"])
        # print("target", sample["target"])
        obs = jnp.transpose(sample["obs"], (1, 0, 2))
        target = jnp.transpose(sample["target"], (1, 0, 2))

        # print("train sample", obs.shape)
        # print("train target", target.shape)

        # assert (sample["obs"][:, 1:] == sample["target"][:, -1])

        start = time.time()
        loss, last_loss, self.state = self.train_xmap(self.state, obs, target)
        loss = np.array(loss)
        last_loss = np.array(last_loss)
        # print(f"iter done in {time.time() - start:.06}s")
        return loss.mean(), last_loss.mean()

    def eval(self, sample):
        # print("eval sample", sample["obs"].shape)
        # print("eval target", sample["target"].shape)

        start = time.time()
        loss = self.eval_xmap(self.state, sample["obs"], sample["target"])
        loss = np.array(loss)
        # print(f"eval done in {time.time() - start:.06}s")
        return loss.mean()
