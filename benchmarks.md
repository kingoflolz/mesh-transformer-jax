Note that everything on this page is quite outdated, and are only roughly accurate when considering new features
such as RoPE

# Benchmarks (v3-8)

(see `tpuv38_example.py`):

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

## 10B model
```
Initialized in 152.762s
Total parameters: 10073579776
Compiled in 92.6539s
it: 0, loss: 5.3125
<snip>
it: 40, loss: 3.65625
50 steps in 100.235s
effective flops (not including attn): 2.46988e+14
```

# Benchmarks (v3-32)

(see `eval.py`):

## 6B model
```
"layers": 28,
"d_model": 4096,
"n_heads": 16,
"n_vocab": 50400,

"seq": 2048,
"cores_per_replica": 8,
"per_replica_batch": 1,
"gradient_accumulation_steps": 8,

params: 6053381856
32 iters done in 107.935s, avg 3.37298s, 0.296473/s
effective flops (not including attn): 7.05692e+14
MXU flops: 1.04523e+15
```

## Note that the below models do not currently work
They require a larger degree of model parallelism than is currently implemented, but benchmark numbers should be
reasonably representative.

## 13B model
```
"layers": 28,
"d_model": 6144,
"n_heads": 32,
"n_vocab": 50400,

"seq": 2048,
"cores_per_replica": 16,
"per_replica_batch": 1,
"gradient_accumulation_steps": 16,

params: 13312183008
32 iters done in 250.86s, avg 7.83937s, 0.127561/s
effective flops (not including attn): 6.67727e+14
MXU flops: 9.80066e+14
```

## 23B model
```
"layers": 28,
"d_model": 8192,
"n_heads": 32,
"n_vocab": 50400,

"seq": 2048,
"cores_per_replica": 32,
"per_replica_batch": 1,
"gradient_accumulation_steps": 32,

params: 23398107360
16 iters done in 221.33s, avg 13.8331s, 0.0722902/s
effective flops (not including attn): 6.65107e+14
MXU flops: 9.88548e+14
```