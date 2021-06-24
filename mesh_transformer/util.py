import jax
import jax.numpy as jnp
from optax._src.transform import OptState, GradientTransformation, AdditiveWeightDecayState
import numpy as np

def gpt3_schedule(warmup_steps,
                  total_steps,
                  peak_lr,
                  end_lr):
    def sch(step):
        warmup_pct = jnp.clip(step, 0, warmup_steps) / warmup_steps
        anneal_pct = jnp.clip(step - warmup_steps, 0, total_steps) / total_steps

        return warmup_pct * peak_lr - (peak_lr - end_lr) * (1 - jnp.cos(jnp.pi * anneal_pct)) / 2

    return sch


def global_norm(updates):
    pre_sqrt = sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)])
    pre_sqrt = jax.lax.psum(pre_sqrt, "shard")
    return jnp.sqrt(pre_sqrt)


class ClipByGlobalNormState(OptState):
    """The `clip_by_global_norm` transformation is stateless."""


def clip_by_global_norm(max_norm) -> GradientTransformation:
    """Clip updates using their global norm.

    References:
      [Pascanu et al, 2012](https://arxiv.org/abs/1211.5063)

    Args:
      max_norm: the maximum global norm for an update.

    Returns:
      An (init_fn, update_fn) tuple.
    """

    def init_fn(_):
        return ClipByGlobalNormState()

    def update_fn(updates, state, params=None):
        del params
        g_norm = global_norm(updates)
        trigger = g_norm < max_norm
        updates = jax.tree_map(
            lambda t: jnp.where(trigger, t, (t / g_norm) * max_norm), updates)
        return updates, state

    return GradientTransformation(init_fn, update_fn)


def additive_weight_decay(weight_decay: float = 0.0) -> GradientTransformation:
    """Add parameter scaled by `weight_decay`, to all parameters with more than one dim (i.e. exclude ln, bias etc)

    Args:
      weight_decay: a scalar weight decay rate.

    Returns:
      An (init_fn, update_fn) tuple.
    """

    def init_fn(_):
        return AdditiveWeightDecayState()

    def update_fn(updates, state, params):
        updates = jax.tree_multimap(lambda g, p: g + weight_decay * p * (len(g.shape) > 1), updates, params)
        return updates, state

    return GradientTransformation(init_fn, update_fn)


def to_f32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t)


def to_bf16(t):
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t)


def to_f16(t):
    return jax.tree_map(lambda x: x.astype(jnp.float16) if x.dtype == jnp.float32 else x, t)


# identity in forward pass, psum in backward
@jax.custom_vjp
def f_psum(x):
    return x


def f_psum_fwd(x):
    return f_psum(x), None


def f_psum_bwd(_, g):
    return jax.lax.psum(g, "shard"),


f_psum.defvjp(f_psum_fwd, f_psum_bwd)


# identity in forward pass, pmean in backward
@jax.custom_vjp
def f_pmean(x):
    return x


def f_pmean_fwd(x):
    return f_psum(x), None


def f_pmean_bwd(_, g):
    return jax.lax.pmean(g, "shard"),


f_pmean.defvjp(f_pmean_fwd, f_pmean_bwd)


# psum in forward pass, identity in backward
@jax.custom_vjp
def g_psum(x):
    return jax.lax.psum(x, "shard")


def g_psum_fwd(x):
    return g_psum(x), None


def g_psum_bwd(_, g):
    return g,


g_psum.defvjp(g_psum_fwd, g_psum_bwd)


def shard_axis(x, axis_size, axis_name):
    # in_shape = x.shape
    assert x.shape[0] % axis_size == 0

    x = x.reshape((axis_size, -1) + x.shape[1:])

    x = x[jax.lax.axis_index(axis_name)]
    # print("shard out", x.shape, "in", in_shape)

    # assert np.prod(x.shape) * axis_size == np.prod(in_shape)

    return x


def unshard_axis(x, axis_name):
    # in_shape = x.shape
    x = jax.lax.all_gather(x, axis_name)

    x = x.reshape((-1, ) + x.shape[2:])

    # assert x.shape[-1] == 4096
    # print("unshard out", x.shape, "in", in_shape)
    return x


if __name__ == "__main__":
    sch = gpt3_schedule(1_000, 20_000, 1e-4, 1e-5)

    for i in range(150):
        i = i * 200
        print(i, sch(i))
