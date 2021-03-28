import jax
import jax.numpy as jnp
import haiku as hk


# takes in a logit distribution, softmax and then sample
def softmax_sample(probs, _):
    return jax.random.categorical(hk.next_rng_key(), probs, -1).astype(jnp.uint32), None
