import jax
import jax.numpy as jnp
import haiku as hk


# takes in a logit distribution, softmax and then sample
def softmax_sample(key, logits, _):
    return jax.random.categorical(key, logits, -1).astype(jnp.uint32), None

# TODO make this actually work
def nucleaus_sample(key, logits, _, top_p=0.9):
    sorted_logits = jnp.sort(logits)[:, ::-1] # sort descending
    sorted_indices = jnp.argsort(logits)[:, ::-1]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits), axis=-1)

    # Remove tokens with cumulative probability above a threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = jnp.concatenate((jnp.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove), axis=-1)[:, :-1]

    indices_to_remove = jax.ops.index_update(jnp.zeros_like(sorted_indices_to_remove), sorted_indices, sorted_indices_to_remove)

    logit_mask = 1e10 * indices_to_remove

    logits -= logit_mask

    return softmax_sample(key, logits, None)
