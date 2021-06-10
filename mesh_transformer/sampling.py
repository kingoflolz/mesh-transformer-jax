import jax
import jax.numpy as jnp


# takes in a logit distribution, softmax and then sample
def softmax_sample(key, logits, _, temp=1):
    return jax.random.categorical(key, logits/temp, -1).astype(jnp.uint32), None


def nucleaus_filter(logits, top_p=0.9, top_k=None):
    sorted_logits = jnp.sort(logits)[:, ::-1] # sort descending
    sorted_indices = jnp.argsort(logits)[:, ::-1]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits), axis=-1)

    if top_k is not None:
        # Keep only top_k tokens
        indices_range = jnp.arange(len(sorted_indices[0]))
        indices_range = jnp.stack([indices_range] * len(sorted_indices), axis=0)

        sorted_indices_to_remove = jnp.where(indices_range > top_k, sorted_indices, 0)

        _, indices_to_remove = jax.lax.sort_key_val(sorted_indices, sorted_indices_to_remove)

        logit_mask = 1e10 * indices_to_remove

        logits -= logit_mask

    # Remove tokens with cumulative probability above a threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = jnp.concatenate((jnp.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove), axis=-1)[:, :-1]

    _, indices_to_remove = jax.lax.sort_key_val(sorted_indices, sorted_indices_to_remove)

    logit_mask = 1e10 * indices_to_remove

    logits -= logit_mask

    return logits


def nucleaus_sample(key, logits, _, top_p=0.9, temp=1, top_k=None):
    logits = nucleaus_filter(logits, top_p, top_k=top_k)

    return softmax_sample(key, logits, None, temp=temp)


if __name__ == "__main__":
    import numpy as np
    logits = np.array([[-2, -1, 0, 0.8, 0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, -3]])
    print(nucleaus_filter(logits))