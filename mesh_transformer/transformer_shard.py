import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.maps import thread_resources


def to_f32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t)


def to_bf16(t):
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t)


class RelativePositionEmbs(hk.Module):
    @staticmethod
    def _relative_position_bucket(relative_position,
                                  num_buckets=32,
                                  max_distance=128):
        ret = 0
        n = -relative_position
        n = np.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = (n < max_exact)
        val_if_large = max_exact + (
                np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
                np.log(max_distance / max_exact) *
                (num_buckets - max_exact)).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        ret += np.where(is_small, n, val_if_large)
        return ret

    def __call__(self, qlen, klen, heads, num_buckets):
        """Produce relative position embedding attention biases.
        Returns:
          output: `(heads, q_len, k_len)` attention bias
        """
        context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
        memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(relative_position)
        relative_attention_bias = hk.get_parameter('rel_embedding', [heads, num_buckets], init=hk.initializers.TruncatedNormal(stddev=0.02))
        # Instead of using a slow gather, we create a leading-dimension one-hot
        # array from rp_bucket and use it to perform the gather-equivalent via a
        # contraction, i.e.:
        # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
        # This is equivalent to relative_attention_bias[:, rp_bucket]
        bcast_iota = jax.lax.broadcasted_iota(jnp.int32, (num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, Ellipsis] == bcast_iota).astype(relative_attention_bias.dtype)
        # --> shape (qlen, klen, num_heads)
        values = jax.lax.dot_general(
            relative_attention_bias,
            rp_bucket_one_hot,
            (
                ((1,), (0,)),  # rhs, lhs contracting dims
                ((), ())))  # no batched dims
        return values


class EmbeddingShard(hk.Module):
    def __init__(self, in_dim, out_dim, shards, name=None):
        super().__init__(name=name)
        assert in_dim % shards == 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dim_per_shard = in_dim // shards
        self.shards = shards

        self.out_dim_per_shard = out_dim // shards

        # embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        # self.positional_embeddings = hk.get_parameter('pos_embs', [seq_length, self.out_dim_per_shard], init=embed_init)
        self.proj = hk.Linear(self.out_dim, w_init=hk.initializers.TruncatedNormal(stddev=1 / np.sqrt(in_dim)))

    def __call__(self, x, dtype=jnp.bfloat16):
        shard_start_index = jax.lax.axis_index('shard') * self.in_dim_per_shard
        shard_index = jnp.arange(0, self.in_dim_per_shard) + shard_start_index

        proj_out = self.proj((shard_index.reshape(1, -1) == x.reshape(-1, 1)).astype(dtype))

        # all_pos_embed = jax.lax.all_gather(self.positional_embeddings, 'shard')
        # all_pos_embed = hk.Flatten()(jnp.transpose(all_pos_embed, (1, 0, 2)))

        return jax.lax.pmean(proj_out, "shard")  # + all_pos_embed


# We actually combine the FF and dense in one layer (i.e. compute in parallel) to minimize all reduces
class TransformerLayerShard(hk.Module):
    def __init__(self, dim, heads, shards, init_scale=1., name=None):
        super().__init__(name=name)
        assert dim % heads == 0
        assert heads % shards == 0

        self.dim = dim
        self.dim_per_head = dim // heads
        self.heads_per_shard = heads // shards
        self.dim_per_shard = dim // shards

        self.ln = hk.LayerNorm(-1, True, True)

        self.q = hk.Linear(self.dim_per_shard, with_bias=False)
        self.v = hk.Linear(self.dim_per_shard, with_bias=False)
        self.k = hk.Linear(self.dim_per_shard, with_bias=False)

        self.o = hk.Linear(self.dim, with_bias=False,
                           w_init=hk.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(self.dim)))

        self.dense_proj = hk.Linear(self.dim_per_shard * 4)
        self.dense_proj_o = hk.Linear(self.dim,
                                      w_init=hk.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(self.dim)))

    def __call__(self, x, attn_bias):
        x = self.ln(x)

        q = self.q(x).reshape((-1, self.heads_per_shard, self.dim_per_head))
        v = self.v(x).reshape((-1, self.heads_per_shard, self.dim_per_head))
        k = self.k(x).reshape((-1, self.heads_per_shard, self.dim_per_head))

        attention_logits = jnp.einsum("thd,Thd->htT", q, k)

        sqrt_key_size = np.sqrt(self.dim_per_head).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size

        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        attention_logits -= 1e10 * (1. - causal_mask)

        if attn_bias is not None:
            attention_logits += attn_bias

        attention_weights = jax.nn.softmax(attention_logits)
        attention_vec = jnp.einsum("htT,Thd->thd", attention_weights, v).reshape((-1, self.dim_per_shard))

        attn_out = self.o(attention_vec)

        dense_proj = self.dense_proj(x)
        dense_proj = jax.nn.gelu(dense_proj)
        dense_out = self.dense_proj_o(dense_proj)

        return jax.lax.pmean(attn_out + dense_out, "shard")


class ProjectionShard(hk.Module):
    def __init__(self, out_dim, shards, name=None):
        super().__init__(name=name)
        assert out_dim % shards == 0

        self.dim = out_dim
        self.dim_per_shard = out_dim // shards
        self.shards = shards

        self.ln = hk.LayerNorm(-1, True, True)

        self.proj = hk.Linear(self.dim_per_shard)

    def __call__(self, x):
        x = self.ln(x)
        proj = self.proj(x)

        all_proj = jax.lax.all_gather(proj, 'shard')

        return hk.Flatten()(jnp.transpose(all_proj, (1, 0, 2)))

    def loss(self, x, targets, dtype=jnp.bfloat16):
        shard_logits = self.proj(x)

        shard_start_index = jax.lax.axis_index('shard') * self.dim_per_shard
        shard_index = jnp.arange(0, self.dim_per_shard) + shard_start_index

        gt_onehot = (shard_index.reshape(1, -1) == targets.reshape(-1, 1)).astype(dtype)

        shifted = shard_logits - jax.lax.stop_gradient(
            jax.lax.pmax(jax.lax.stop_gradient(shard_logits.max(-1, keepdims=True)), "shard"))
        logsoftmax = shifted - jnp.log(jax.lax.psum(jnp.sum(jnp.exp(shifted), -1, keepdims=True), "shard"))

        loss = jax.lax.psum(-jnp.sum(gt_onehot * logsoftmax, axis=-1), "shard")

        return loss


class CausalTransformerShard(hk.Module):
    def __init__(self, dim, heads, layer_count, seq, vocab):
        super().__init__()
        self.transformer_layers = []
        self.heads = heads

        shards = thread_resources.env.shape['mp']
        self.heads_per_shard = heads // shards

        self.embed = EmbeddingShard(vocab, dim, shards)

        init_scale = 2. / layer_count

        for i in range(layer_count):
            self.transformer_layers.append(
                TransformerLayerShard(dim, heads, shards, init_scale=init_scale, name=f"layer_{i}"))

        self.proj = ProjectionShard(vocab, shards)
        self.rpe = RelativePositionEmbs()

    def eval(self, context, target):
        input_len = context.shape[0]

        attn_bias = self.rpe(input_len, input_len, self.heads_per_shard, 32)
        # attn_bias = hk.get_parameter('rel_embedding', [self.heads_per_shard, input_len, input_len], init=hk.initializers.TruncatedNormal(stddev=1 / input_len))

        x = hk.remat(self.embed)(context)

        for l in self.transformer_layers:
            x = x + hk.remat(l)(x, attn_bias)

        return hk.remat(self.proj.loss)(x, target)

    def train_loss(self, ctx, tgt):
        return self.eval(ctx, tgt).mean()


class CausalTransformer:
    def __init__(self, dim: int, heads: int, layer_count: int, vocab: int, seq: int,
                 optimizer: optax.GradientTransformation,
                 deterministic: bool = True):
        self.heads = heads

        def eval(state, ctx, tgt):
            def eval_loss(x, y):
                transformer = CausalTransformerShard(dim, heads, layer_count, seq, vocab)
                return transformer.eval(x, y)

            eval_loss_fn = hk.without_apply_rng(hk.transform(eval_loss)).apply(ctx, tgt)

            return eval_loss_fn(to_bf16(state["params"]), ctx, tgt)

        def train(state, ctx, tgt):
            def train_loss(x, y):
                transformer = CausalTransformerShard(dim, heads, layer_count, seq, vocab)
                return transformer.train_loss(x, y)

            train_loss_fn = hk.without_apply_rng(hk.transform(train_loss)).apply

            value, grad = jax.value_and_grad(train_loss_fn)(to_bf16(state["params"]), ctx, tgt)
            grad = jax.lax.pmean(grad, "batch")
            updates, new_opt_state = optimizer.update(grad, state["opt_state"])

            return to_f32(value), {
                "params": optax.apply_updates(state["params"], to_f32(updates)),
                "step": state["step"] + 1,
                "opt_state": new_opt_state,
            }

        def init(key, x):
            def train_loss(x, y):
                transformer = CausalTransformerShard(dim, heads, layer_count, seq, vocab)
                return transformer.train_loss(x, y)

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
                                                     out_axes=(["batch", ...], ["shard", ...]),
                                                     donate_argnums=(0,),
                                                     axis_resources={'shard': 'mp', 'batch': 'dp'})

        if deterministic:
            # key = hk.PRNGSequence(42 + jax.host_id())
            key = hk.PRNGSequence(42)  # we actually need this to be completely deterministic until xmap is fixed
        else:
            # key = hk.PRNGSequence(random.randrange(1e9))
            key = hk.PRNGSequence(42)

        dp = thread_resources.env.shape['dp']
        mp = thread_resources.env.shape['mp']
        x = jax.random.uniform(next(key), (dp//jax.host_count(), seq,), minval=0, maxval=vocab).astype(jnp.int32)  # batch, len

        print("key shape", jnp.array(key.take(mp)).shape)
        print("in shape", x.shape)

        print("dp", dp)
        print("mp", mp)

        self.state = self.init_xmap(jnp.array(key.take(mp)), x)

    def train(self, sample):
        # print("train iter")
        # print("sample", sample["obs"])
        # print("target", sample["target"])

        # assert (sample["obs"][:, 1:] == sample["target"][:, -1])

        start = time.time()
        loss, self.state = self.train_xmap(self.state, sample["obs"], sample["target"])
        loss = np.array(loss)
        # print(f"iter done in {time.time() - start:.06}s")
        return loss.mean()
