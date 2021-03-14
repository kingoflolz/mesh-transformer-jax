# Mesh Transformer Jax

A haiku library using the new(ly documented) `xmap` operator in Jax for model parallelism of transformers.

See `enwik8_example.py` for an example of using this to implement an autoregressive language model.

# Benchmarks

On a TPU v3-8 (see `tpuv38_example.py`):

```
Initialized in 121.842s
Total parameters: 2722382080
Compiled in 49.0534s
it: 0, loss: 20.311113357543945
<snip>
it: 90, loss: 3.987450361251831
100 steps in 109.385s
effective flops (not including attn): 2.4466e+14
```

# TODO
- [x] disentangle heads and shards
- [x] test/benchmark on TPU
- [ ] implement gradient checkpointing
- [ ] test groupnorm vs layernorm
- [ ] shard activations instead of replicating