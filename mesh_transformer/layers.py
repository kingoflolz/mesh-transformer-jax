import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat

from mesh_transformer.util import f_psum, g_psum


class ReplicatedLayerNorm(hk.Module):
    def __init__(self, offset=True):
        super().__init__()
        self.offset = offset

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.var(inputs, axis=-1, keepdims=True)

        param_shape = inputs.shape[-1:]
        scale = hk.get_parameter("scale", param_shape, inputs.dtype, init=jnp.ones)
        scale = jax.lax.all_gather(scale, "shard")[0]

        offset = hk.get_parameter("offset", param_shape, inputs.dtype, init=jnp.zeros)
        offset = jax.lax.all_gather(offset, "shard")[0]

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        inv = scale * jax.lax.rsqrt(variance + 1e-5)
        if self.offset:
            return inv * (inputs - mean) + offset
        else:
            return inv * (inputs - mean)


class RMSNorm(hk.Module):
    def __init__(self, offset, elementwise):
        super().__init__()
        self.offset = offset
        self.elementwise = elementwise

    def __call__(self, x):
        param_shape = (x.shape[-1],) if self.elementwise else ()
        normed = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-5)

        scale = hk.get_parameter('scale', param_shape, init=hk.initializers.Constant(x.shape[-1] ** 0.5))
        scale = jax.lax.pmean(scale, "shard")
        normed = normed * scale

        if self.offset:
            offset = hk.get_parameter('offset', param_shape, init=jnp.zeros)
            offset = jax.lax.pmean(offset, "shard")
            normed = normed + offset

        return normed


def getnorm(type):
    if type == "layernorm":
        return ReplicatedLayerNorm()
    if type == "layernorm-desync":
        return hk.LayerNorm(-1, True, True)
    elif type == "layernorm-nobias":
        return ReplicatedLayerNorm(offset=False)
    elif type == "rmsnorm":
        return RMSNorm(False, True)
    elif type == "scalenorm":
        return RMSNorm(False, False)
    elif type == "rmsnorm-bias":
        return RMSNorm(True, True)
    elif type == "scalenorm-bias":
        return RMSNorm(True, False)
    else:
        raise Exception("Not implemented")


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
        relative_attention_bias = hk.get_parameter('rel_embedding', [heads, num_buckets],
                                                   init=hk.initializers.TruncatedNormal(stddev=0.02))
        # Instead of using a slow gather, we create a leading-dimension one-hot
        # array from rp_bucket and use it to perform the gather-equivalent via a
        # contraction, i.e.:
        # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
        # This is equivalent to relative_attention_bias[:, rp_bucket]
        bcast_iota = jax.lax.broadcasted_iota(jnp.int32, (num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, Ellipsis] == bcast_iota).astype(
            relative_attention_bias.dtype)
        # --> shape (qlen, klen, num_heads)
        values = jax.lax.dot_general(
            relative_attention_bias,
            rp_bucket_one_hot,
            (
                ((1,), (0,)),  # rhs, lhs contracting dims
                ((), ())))  # no batched dims
        return values


def fixed_pos_embedding(x, seq_dim=0):
    dim = x.shape[-1]
    inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim))

    sinusoid_inp = np.einsum('i , j -> i j', np.arange(x.shape[seq_dim]), inv_freq)

    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j=2)[-x.shape[0]:, None, :], sincos)
    return (x * cos) + (rotate_every_two(x) * sin)


class EmbeddingShard(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        in_dim = config["n_vocab"]
        out_dim = config["d_model"]
        shards = config["cores_per_replica"]

        assert in_dim % shards == 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dim_per_shard = in_dim // shards
        self.out_dim_per_shard = out_dim // shards

        if config["pe"] == "fixed":
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            self.positional_embeddings = hk.get_parameter('pos_embs', [config["seq"], self.out_dim_per_shard], init=embed_init)
        else:
            self.positional_embeddings = None

        self.proj = hk.Linear(self.out_dim, w_init=hk.initializers.TruncatedNormal(stddev=1 / np.sqrt(in_dim)))

    def __call__(self, x, dtype=jnp.bfloat16):
        shard_start_index = jax.lax.axis_index('shard') * self.in_dim_per_shard

        input_onehot = jax.nn.one_hot(x - shard_start_index, self.in_dim_per_shard)
        proj_out = self.proj(input_onehot)

        proj_out = g_psum(proj_out)

        if self.positional_embeddings is not None:
            all_pos_embed = jax.lax.all_gather(self.positional_embeddings, 'shard')

            all_pos_embed = hk.Flatten()(jnp.transpose(all_pos_embed, (1, 0, 2)))

            proj_out += all_pos_embed

        return proj_out


# We actually combine the FF and dense in one layer (i.e. compute in parallel) to minimize all reduces
class TransformerLayerShard(hk.Module):
    def __init__(self, config, name=None, init_scale=1.):
        super().__init__(name=name)
        heads = config["n_heads"]
        dim = config["d_model"]
        shards = config["cores_per_replica"]
        norm = getnorm(config["norm"])
        self.is_rotary = config["pe"] == "rotary"

        assert dim % heads == 0
        assert heads % shards == 0

        self.dim = dim
        self.dim_per_head = dim // heads
        self.heads_per_shard = heads // shards
        self.dim_per_shard = dim // shards
        self.pe_rotary_dims = config.get("pe_rotary_dims", self.dim_per_head)

        self.norm = norm

        self.q = hk.Linear(self.dim_per_shard, with_bias=False)
        self.v = hk.Linear(self.dim_per_shard, with_bias=False)
        self.k = hk.Linear(self.dim_per_shard, with_bias=False)

        self.o = hk.Linear(self.dim, with_bias=False,
                           w_init=hk.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(self.dim)))

        self.dense_proj = hk.Linear(self.dim_per_shard * 4)
        self.dense_proj_o = hk.Linear(self.dim,
                                      w_init=hk.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(self.dim)))

    def self_attn(self, q, v, k, attn_bias):
        if self.is_rotary:
            k_rot = k[:, :, :self.pe_rotary_dims]
            k_pass = k[:, :, self.pe_rotary_dims:]

            q_rot = q[:, :, :self.pe_rotary_dims]
            q_pass = q[:, :, self.pe_rotary_dims:]

            sincos = fixed_pos_embedding(k_rot)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)
            k_rot = apply_rotary_pos_emb(k_rot, sincos)

            k = jnp.concatenate([k_rot, k_pass], axis=-1)
            q = jnp.concatenate([q_rot, q_pass], axis=-1)

        attention_logits = jnp.einsum("thd,Thd->htT", q, k)

        sqrt_key_size = np.sqrt(self.dim_per_head).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size

        attention_logits += attn_bias

        attention_weights = jax.nn.softmax(attention_logits)
        attention_vec = jnp.einsum("htT,Thd->thd", attention_weights, v).reshape((-1, self.dim_per_shard))

        return self.o(attention_vec)

    def ff(self, x):
        dense_proj = self.dense_proj(x)
        dense_proj = jax.nn.gelu(dense_proj)
        return self.dense_proj_o(dense_proj)

    def qvk_proj(self, x):
        q = self.q(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
        v = self.v(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
        k = self.k(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))

        return q, v, k

    def __call__(self, x, attn_bias):
        x = f_psum(x)
        x = self.norm(x)

        q, v, k = self.qvk_proj(x)

        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        bias = -1e10 * (1. - causal_mask)
        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        dense_out = self.ff(x)

        return g_psum(attn_out + dense_out)

    # iterate the decoding process by a single token
    def decode_once(self, decode_state, x, attn_bias):
        x = f_psum(x)
        x = self.norm(x)

        assert x.shape[0] == 1

        q, v, k = self.qvk_proj(x)

        # add new kv to end
        v = jnp.concatenate((decode_state["v"], v), axis=0)[1:]
        k = jnp.concatenate((decode_state["k"], k), axis=0)[1:]

        tokens_decoded = decode_state["tokens_decoded"] + 1
        length = v.shape[0]

        masked_tokens = length - tokens_decoded

        attention_mask = jnp.arange(0, length) < masked_tokens
        bias = (-1e10 * attention_mask)
        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        dense_out = self.ff(x)

        return g_psum(attn_out + dense_out), {
            "tokens_decoded": tokens_decoded,
            "k": k,
            "v": v
        }

    # take in right aligned context tokens and generate an initial state
    def get_init_decode_state(self, x, given_length, attn_bias):
        x = f_psum(x)
        x = self.norm(x)

        q, v, k = self.qvk_proj(x)

        full_length = x.shape[0]
        masked_tokens = full_length - given_length

        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))

        bias = -1e10 * (1. - causal_mask)  # regular AR masking
        bias -= 1e10 * (jnp.arange(0, full_length) < masked_tokens)  # mask out zero tokens before context starts
        bias += attn_bias  # finally add attn bias for rpe

        attn_out = self.self_attn(q, v, k, bias)
        dense_out = self.ff(x)

        return g_psum(attn_out + dense_out), {"k": k, "v": v, "tokens_decoded": given_length.astype(jnp.uint32)}


# This new class combines the input and output projection into one matmul for better efficiency
class TransformerLayerShardV2(hk.Module):
    def __init__(self, config, name=None, init_scale=1.):
        super().__init__(name=name)
        heads = config["n_heads"]
        dim = config["d_model"]
        shards = config["cores_per_replica"]
        norm = getnorm(config["norm"])
        self.is_rotary = config["pe"] == "rotary"

        assert dim % heads == 0
        assert heads % shards == 0

        self.dim = dim
        self.dim_per_head = dim // heads
        self.heads_per_shard = heads // shards
        self.dim_per_shard = dim // shards
        self.pe_rotary_dims = config.get("pe_rotary_dims", self.dim_per_head)

        self.norm = norm

        self.input_proj = hk.Linear(self.dim_per_shard * 3 + self.dim_per_shard * 4)
        self.output_proj = hk.Linear(self.dim, w_init=hk.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(self.dim)))

    def self_attn(self, q, v, k, attn_bias):
        if self.is_rotary:
            k_rot = k[:, :, :self.pe_rotary_dims]
            k_pass = k[:, :, self.pe_rotary_dims:]

            q_rot = q[:, :, :self.pe_rotary_dims]
            q_pass = q[:, :, self.pe_rotary_dims:]

            sincos = fixed_pos_embedding(k_rot)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)
            k_rot = apply_rotary_pos_emb(k_rot, sincos)

            k = jnp.concatenate([k_rot, k_pass], axis=-1)
            q = jnp.concatenate([q_rot, q_pass], axis=-1)

        attention_logits = jnp.einsum("thd,Thd->htT", q, k)

        sqrt_key_size = np.sqrt(self.dim_per_head).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size

        attention_logits += attn_bias

        attention_weights = jax.nn.softmax(attention_logits)
        attention_vec = jnp.einsum("htT,Thd->thd", attention_weights, v).reshape((-1, self.dim_per_shard))

        return attention_vec

    def input(self, x):
        q, v, k, ff = jnp.split(self.input_proj(x), np.cumsum([self.dim_per_shard] * 3), axis=-1)

        q = q.reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
        v = v.reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
        k = k.reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))

        return q, v, k, ff

    def output(self, *x):
        return self.output_proj(jnp.concatenate(x, axis=-1))

    def __call__(self, x, attn_bias):
        x = f_psum(x)
        x = self.norm(x)

        q, v, k, ff = self.input(x)

        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        bias = -1e10 * (1. - causal_mask)
        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        ff_out = jax.nn.gelu(ff)

        return g_psum(self.output(attn_out, ff_out))

    # iterate the decoding process by a single token
    def decode_once(self, decode_state, x, attn_bias):
        x = f_psum(x)
        x = self.norm(x)

        assert x.shape[0] == 1

        q, v, k, ff = self.input(x)

        # add new kv to end
        v = jnp.concatenate((decode_state["v"], v), axis=0)[1:]
        k = jnp.concatenate((decode_state["k"], k), axis=0)[1:]

        tokens_decoded = decode_state["tokens_decoded"] + 1
        length = v.shape[0]

        masked_tokens = length - tokens_decoded

        attention_mask = jnp.arange(0, length) < masked_tokens
        bias = (-1e10 * attention_mask)
        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        ff_out = jax.nn.gelu(ff)

        return g_psum(self.output(attn_out, ff_out)), {
            "tokens_decoded": tokens_decoded,
            "k": k,
            "v": v
        }

    # take in right aligned context tokens and generate an initial state
    def get_init_decode_state(self, x, given_length, attn_bias):
        x = f_psum(x)
        x = self.norm(x)

        q, v, k, ff = self.input(x)

        full_length = x.shape[0]
        masked_tokens = full_length - given_length

        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))

        bias = -1e10 * (1. - causal_mask)  # regular AR masking
        bias -= 1e10 * (jnp.arange(0, full_length) < masked_tokens)  # mask out zero tokens before context starts
        bias += attn_bias  # finally add attn bias for rpe

        attn_out = self.self_attn(q, v, k, bias)
        ff_out = jax.nn.gelu(ff)

        return g_psum(self.output(attn_out, ff_out)),\
               {"k": k, "v": v, "tokens_decoded": given_length.astype(jnp.uint32)}


class ProjectionShard(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        out_dim = config["n_vocab"]
        shards = config["cores_per_replica"]
        norm = getnorm(config["norm"])

        assert out_dim % shards == 0

        self.dim = out_dim
        self.dim_per_shard = out_dim // shards

        self.norm = norm

        self.proj = hk.Linear(self.dim_per_shard)

    def __call__(self, x):
        x = self.norm(x)
        proj = self.proj(x)

        all_proj = jax.lax.all_gather(proj, 'shard')

        return hk.Flatten()(jnp.transpose(all_proj, (1, 0, 2)))

    def loss(self, x, targets, z_loss=1):
        x = f_psum(x)
        x = self.norm(x)
        logits = self.proj(x)

        shard_start_index = jax.lax.axis_index('shard') * self.dim_per_shard
        global_max = jax.lax.pmax(jax.lax.stop_gradient(logits.max(-1, keepdims=True)), "shard")
        logits -= jax.lax.stop_gradient(global_max)

        gt_onehot = jax.nn.one_hot(targets - shard_start_index, self.dim_per_shard)
        predicted_logits = jnp.sum(jnp.multiply(gt_onehot, logits), axis=-1)
        predicted_logits = g_psum(predicted_logits)

        exp_logits = jnp.exp(logits)

        sum_exp_logits = exp_logits.sum(axis=-1)
        sum_exp_logits = g_psum(sum_exp_logits)

        loss = jnp.log(sum_exp_logits) - predicted_logits

        loss += (1e-4 * jnp.square(jnp.log(sum_exp_logits)) * z_loss).mean()

        correct = (0.0 == predicted_logits)

        return loss, correct
