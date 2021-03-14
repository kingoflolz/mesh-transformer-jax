import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.maps import thread_resources


class EmbeddingShard(hk.Module):
    def __init__(self, in_dim, out_dim, shards, name=None):
        super().__init__(name=name)
        assert in_dim % shards == 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_per_shard = in_dim // shards
        self.shards = shards

        self.ln = hk.LayerNorm(-1, True, True)

        self.proj = hk.Linear(self.out_dim)

    def __call__(self, x):
        shard_start_index = jax.lax.axis_index('shard') * self.dim_per_shard
        shard_index = jnp.arange(0, self.dim_per_shard) + shard_start_index

        proj_out = self.proj((shard_index.reshape(1, -1) == x.reshape(-1, 1)).astype(jnp.float32))

        return jax.lax.pmean(proj_out, "shard")


# We actually combine the FF and dense in one layer (i.e. compute in parallel) to minimize all reduces
class TransformerLayerShard(hk.Module):
    def __init__(self, dim, heads, shards, name=None):
        super().__init__(name=name)
        assert dim % heads == 0
        assert heads % shards == 0

        self.dim = dim
        self.dim_per_head = dim // heads
        self.heads_per_shard = heads // shards
        self.dim_per_shard = dim // shards

        self.ln = hk.LayerNorm(-1, True, True)

        self.q = hk.Linear(self.dim_per_shard)
        self.v = hk.Linear(self.dim_per_shard)
        self.k = hk.Linear(self.dim_per_shard)

        self.o = hk.Linear(self.dim)

        self.dense_proj = hk.Linear(self.dim_per_head * 4)
        self.dense_proj_o = hk.Linear(self.dim)

    def __call__(self, x, mask=None):
        x = self.ln(x)

        q = self.q(x).reshape((-1, self.heads_per_shard, self.dim_per_head))
        v = self.v(x).reshape((-1, self.heads_per_shard, self.dim_per_head))
        k = self.k(x).reshape((-1, self.heads_per_shard, self.dim_per_head))

        attention_logits = jnp.einsum("thd,Thd->htT", q, k)

        sqrt_key_size = np.sqrt(self.dim_per_head).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size

        if mask is None:
            attention_logits += mask

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

    def loss(self, x, targets):
        shard_logits = self.proj(x)

        shard_start_index = jax.lax.axis_index('shard') * self.dim_per_shard
        shard_index = jnp.arange(0, self.dim_per_shard) + shard_start_index

        gt_onehot = (shard_index.reshape(1, -1) == targets.reshape(-1, 1)).astype(jnp.float32)

        shifted = shard_logits - jax.lax.stop_gradient(jax.lax.pmax(jax.lax.stop_gradient(shard_logits.max(-1, keepdims=True)), "shard"))
        logsoftmax = shifted - jnp.log(jax.lax.psum(jnp.sum(jnp.exp(shifted), -1, keepdims=True), "shard"))

        loss = jax.lax.psum(-jnp.sum(gt_onehot * logsoftmax, axis=-1), "shard")

        return loss


class CausalTransformerShard(hk.Module):
    def __init__(self, dim, heads, layer_count, vocab):
        super().__init__()
        self.transformer_layers = []
        self.heads = heads

        shards = thread_resources.env.shape['mp']

        self.embed = EmbeddingShard(vocab, dim, shards)

        for i in range(layer_count):
            self.transformer_layers.append(TransformerLayerShard(dim, heads, shards))

        self.proj = ProjectionShard(vocab, shards)

    def eval(self, context, target):
        x = self.embed(context)

        mask = jnp.zeros((x.shape[0], x.shape[0]))
        mask -= 10e10
        mask = jnp.triu(mask, 1)  # zero out the lower diagonal

        for l in self.transformer_layers:
            x = x + l(x, mask)

        return self.proj.loss(x, target)

    def train_loss(self, ctx, tgt):
        return self.eval(ctx, tgt).mean()


class CausalTransformer:
    def __init__(self, dim: int, heads: int, layer_count: int, vocab: int, optimizer):
        self.heads = heads

        def eval(state, ctx, tgt):
            def eval_loss(x, y):
                transformer = CausalTransformerShard(dim, heads, layer_count, vocab)
                return transformer.eval(x, y)

            eval_loss_fn = hk.without_apply_rng(hk.transform(eval_loss)).apply(ctx, tgt)

            return eval_loss_fn(state["params"], ctx, tgt)

        def train(state, ctx, tgt):
            def train_loss(x, y):
                transformer = CausalTransformerShard(dim, heads, layer_count, vocab)
                return transformer.train_loss(x, y)

            train_loss_fn = hk.without_apply_rng(hk.transform(train_loss)).apply

            value, grad = jax.value_and_grad(train_loss_fn)(state["params"], ctx, tgt)
            grad = jax.lax.pmean(grad, "batch")

            updates, new_opt_state = optimizer.update(grad, state["opt_state"])

            return value, {
                "params": optax.apply_updates(state["params"], updates),
                "step": state["step"] + 1,
                "opt_state": new_opt_state
            }

        def init(key, x):
            def train_loss(x, y):
                transformer = CausalTransformerShard(dim, heads, layer_count, vocab)
                return transformer.train_loss(x, y)

            param_init_fn = hk.transform(train_loss).init

            params = param_init_fn(key, x, x)

            return {
                "params": params,
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

        key = hk.PRNGSequence(42)

        dp = thread_resources.env.shape['dp']
        mp = thread_resources.env.shape['mp']
        x = jax.random.uniform(next(key), (dp, 64,), minval=0, maxval=vocab).astype(jnp.int32)  # batch, len

        self.state = self.init_xmap(jnp.array(key.take(mp)), x)

    def train(self, sample):
        loss, self.state = self.train_xmap(self.state, sample["obs"], sample["target"])
        return loss
