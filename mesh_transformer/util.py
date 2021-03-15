import optax
import jax.numpy as jnp

def gpt3_schedule(warmup_steps,
                  total_steps,
                  peak_lr,
                  end_lr):
    
    def sch(step):
        warmup_pct = jnp.clip(step, 0, warmup_steps) / warmup_steps
        anneal_pct = jnp.clip(step - warmup_steps, 0, total_steps) / total_steps

        return warmup_pct * peak_lr - (peak_lr - end_lr) * (1 - jnp.cos(jnp.pi * anneal_pct)) / 2
    
    return sch

if __name__ == "__main__":
    sch = gpt3_schedule(1_000, 20_000, 1e-4, 1e-5)

    for i in range(150):
        i = i * 200
        print(i, sch(i))