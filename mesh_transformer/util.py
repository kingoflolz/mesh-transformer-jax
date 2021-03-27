import jax
import jax.numpy as jnp
from optax._src.transform import OptState, GradientTransformation


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


if __name__ == "__main__":
    sch = gpt3_schedule(1_000, 20_000, 1e-4, 1e-5)

    for i in range(150):
        i = i * 200
        print(i, sch(i))
