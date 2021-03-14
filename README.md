# Mesh Transformer Jax

A haiku library using the new(ly documented) `xmap` operator in Jax for model parallelism of transformers.

See `enwik8_example.py` for an example of using this to implement an autoregressive language model.

# TODO
- [x] disentangle heads and shards
- [ ] test/benchmark on TPU
- [ ] implement gradient checkpointing
- [ ] shard activations instead of replicating