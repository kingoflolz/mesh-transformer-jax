# Mesh Transformer Jax

A haiku library using the new(ly documented) `xmap` operator in Jax for model parallelism of transformers.

See `enwik8_example.py` for an example of using this to implement an autoregressive language model.

# Benchmarks

On a TPU v3-8 (see `tpuv38_example.py`):

## ~2.7B model
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

## ~4.8B model
```
Initialized in 101.016s
Total parameters: 4836720896
Compiled in 52.7404s
it: 0, loss: 4.632925987243652
<snip>
it: 40, loss: 3.2406811714172363
50 steps in 102.559s
effective flops (not including attn): 2.31803e+14
```

# TODO
- [x] disentangle heads and shards
- [x] test/benchmark on TPU
- [x] implement gradient checkpointing
- [x] fix initialization
- [ ] mixed precision
- [ ] shard activations instead of replicating