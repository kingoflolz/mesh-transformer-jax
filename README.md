# Table of contents
1. [Mesh Transformer JAX](#mesh-transformer-jax)
    1. [Updates](#updates)
2. [Pretrained Models](#pretrained-models)
   1. [GPT-J-6B](#gpt-j-6b)
      1. [Links](#links)
      2. [Acknowledgments](#acknowledgments)
      3. [License](#license)
      4. [Model Details](#model-details)
      5. [Zero-Shot Evaluations](#zero-shot-evaluations)
3. [Architecture and Usage](#architecture-and-usage)
   1. [Fine-tuning](#fine-tuning)
   2. [JAX Dependency](#jax-dependency)
4. [TODO](#todo)

# Mesh Transformer JAX

A haiku library using the `xmap`/`pjit` operators in JAX for model parallelism of transformers.

The parallelism scheme is similar to the [original Megatron-LM](https://arxiv.org/abs/1909.08053), which is efficient
on TPUs due to the high speed 2d mesh network. There is also an experimental model version which implements [ZeRo style
sharding](https://arxiv.org/abs/1910.02054).

This library is designed for scalability up to approximately 40B parameters on TPUv3s, beyond which different
parallelism strategies should be used. See other implementations such as
[GPT-NeoX](https://github.com/EleutherAI/gpt-neox) or [DeepSpeed](https://github.com/microsoft/DeepSpeed) for that.

One future direction for research is integrating this codebase with
[swarm-jax](https://github.com/kingoflolz/swarm-jax), to achieve further scalability with pipeline parallelism.

## Updates

**12-07-21**: Added [guide to fine tuning](howto_finetune.md)

# Pretrained Models

## GPT-J-6B

A 6 billion parameter, autoregressive text generation model trained on [The Pile](https://pile.eleuther.ai/).

### Links

[Slim weights (bf16 weights only, for inference, 9GB)](https://the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd)

[Full weights (including optimizer params, 61GB)](https://the-eye.eu/public/AI/GPT-J-6B/step_383500.tar.zstd)

[Colab demo](http://colab.research.google.com/github/kingoflolz/mesh-transformer-jax/blob/master/colab_demo.ipynb)

[Web demo](https://6b.eleuther.ai/)

[Aran's blog post](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/)

### Acknowledgments

This project would not have been possible without compute generously provided by the
[TPU Research Cloud](https://sites.research.google/trc/) with assistance from [EleutherAI](https://eleuther.ai/).

Thanks to the Cloud TPU team at Google for providing early access to the Cloud TPU VM alpha
([now publicly available!](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms))

Thanks to everyone who have helped out one way or another (listed alphabetically):
- [Aran Komatsuzaki](https://twitter.com/arankomatsuzaki) for advice with experiment design and writing the blog posts.
- [James Bradbury](https://twitter.com/jekbradbury) for valuable assistance with debugging JAX issues.
- [Janko Prester](https://github.com/jprester) for creating the web demo frontend.
- [Laurence Golding](https://github.com/researcher2) for adding some features to the web demo.
- [Leo Gao](https://twitter.com/nabla_theta) for running zero shot evaluations for the baseline models for the table.

### License
The weights of GPT-J-6B are licensed under version 2.0 of the Apache License.

### Model Details

| Hyperparameter    | Value  |
|-------------------|--------|
| n_parameters      | 6,053,381,344 |
| n_layers          | 28*    |
| d_model           | 4,096  |
| d_ff              | 16,384 |
| n_heads           | 16     |
| d_head            | 256    |
| n_ctx             | 2,048  |
| n_vocab           | 50,257 (same tokenizer as GPT-2/3)  |
| position encoding | [Rotary position encodings (RoPE)](https://arxiv.org/abs/2104.09864) |
| RoPE dimensions   | [64](https://github.com/kingoflolz/mesh-transformer-jax/blob/f2aa66e0925de6593dcbb70e72399b97b4130482/mesh_transformer/layers.py#L223) |

`*` each layer consists of one feedforward block and one self attention block

The model consists of 28 layers with a model dimension of 4096, and a feedforward dimension of 16384. The model
dimension is split into 16 heads, each with a dimension of 256. Rotary position encodings (RoPE) was applied to 64
dimensions of each head. The model is trained with a tokenization vocabulary of 50257, using the same set of BPEs as
GPT-2/GPT-3.

### Zero-Shot Evaluations

Models roughly sorted by performance, or by FLOPs if not available.

|  Model          | Weights | Training FLOPs | LAMBADA PPL ↓ | LAMBADA Acc ↑ | Winogrande ↑ | Hellaswag ↑ | PIQA ↑ | Dataset Size (GB) |
|-----------------|---------|----------------|---            |---            |---           |---          |---     |-------------------|
| Chance          | ✔       | 0              | ~a lot        | ~0%           | 50%          | 25%         | 25%    | 0                 |
| GPT-3-Ada‡      | ✘       | -----          | 9.95          | 51.6%         | 52.9%        | 43.4%       | 70.5%  | -----             |
| GPT-2-1.5B      | ✔       | -----          | 10.63         | 51.21%        | 59.4%        | 50.9%       | 70.8%  | 40                |
| GPTNeo-1.3B‡    | ✔       | 3.0e21         | 7.50          | 57.2%         | 55.0%        | 48.9%       | 71.1%  | 825               |
| Megatron-2.5B*  | ✘       | 2.4e21         | -----         | 61.7%         | -----        | -----       | -----  | 174               |
| GPTNeo-2.7B‡    | ✔       | 6.8e21         | 5.63          | 62.2%         | 56.5%        | 55.8%       | 73.0%  | 825               |
| GPT-3-1.3B*‡    | ✘       | 2.4e21         | 5.44          | 63.6%         | 58.7%        | 54.7%       | 75.1%  | ~800              |
| GPT-3-Babbage‡  | ✘       | -----          | 5.58          | 62.4%         | 59.0%        | 54.5%       | 75.5%  | -----             |
| Megatron-8.3B*  | ✘       | 7.8e21         | -----         | 66.5%         | -----        | -----       | -----  | 174               |
| GPT-3-2.7B*‡    | ✘       | 4.8e21         | 4.60          | 67.1%         | 62.3%        | 62.8%       | 75.6%  | ~800              |
| Megatron-11B†   | ✔       | 1.0e22         | -----         | -----         | -----        | -----       | -----  | 161               |
| **GPT-J-6B**‡   | ✔       | 1.5e22         | 3.99          | 69.7%         | 65.3%        | 66.1%       | 76.5%  | 825               |
| GPT-3-6.7B*‡    | ✘       | 1.2e22         | 4.00          | 70.3%         | 64.5%        | 67.4%       | 78.0%  | ~800              |
| GPT-3-Curie‡    | ✘       | -----          | 4.00          | 69.3%         | 65.6%        | 68.5%       | 77.9%  | -----             |
| GPT-3-13B*‡     | ✘       | 2.3e22         | 3.56          | 72.5%         | 67.9%        | 70.9%       | 78.5%  | ~800              |
| GPT-3-175B*‡    | ✘       | 3.1e23         | 3.00          | 76.2%         | 70.2%        | 78.9%       | 81.0%  | ~800              |
| GPT-3-Davinci‡  | ✘       | -----          | 3.0           | 75%           | 72%          | 78%         | 80%    | -----             |

`*` represents evaluation numbers reported by their respective authors, all other numbers are provided by
running the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/) either with the released
weights or with API access. Due to subtle implementation differences as well as different zero shot task framing, these
might not be directly comparable. See [this blog post](https://www.eleuther.ai/research-log/gpt3-model-sizes/) for more
details.

`†` The Megatron-11B model provides no comparable metrics, and several implementations using the released weights do not
reproduce the generation quality and evaluations. (see [1](https://github.com/huggingface/transformers/pull/10301)
[2](https://github.com/pytorch/fairseq/issues/2358) [3](https://github.com/pytorch/fairseq/issues/2719))
Thus, evaluation was not attempted.

`‡` These models have been trained with data which contains possible test set contamination. The OpenAI GPT-3 models
failed to deduplicate training data for certain test sets, while the GPT-Neo models as well as this one is
trained on The Pile, which has not been deduplicated against any test sets.

# Architecture and Usage

Most scripts in this repository are designed to be run on TPUs, which under the
[TPU-VM architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) are virtual machines
which can run arbitrary code. Most scripts are designed to spin up a TPU, SSH into it to set up the dependencies
and copy code over from the local directory, and then start a [Ray](https://github.com/ray-project/ray.git) worker
which can accept RPC calls.

The TPUVMs handles running model training steps and evaluation, checkpoint save and loading, while the driver python
program handles data loading and general orchestration (such as when to save checkpoints etc).

This means that most scripts (`train.py`, `eval_harness.py` etc) expect to be running on a GCE virtual machine in the
same region as the TPUs, to minimize RPC latency and data transfer cost. Other scripts
(usually ones which don't take a `--tpu` argument, such as `device_sample.py`, `device_serve.py` or `device_train.py`)
expect to be run directly on a TPUVM. The device_* scripts **only work on a v3-8** and not on larger pods.

Furthermore, there is an example (`resharding_example.py`) of how to convert the provided checkpoints (which have 8
shards in the case of GPT-J-6B) down to a smaller number, such as for when running on GPU(s).

### Fine-tuning

To fine-tune the model, run `device_train.py` on a TPU VM.  Using a TPU v3-8, you can fine-tune at a rate of ~5000
tokens/second, which should be sufficient for small-to-medium-size datasets.

Please read the [step by step guide](howto_finetune.md) for thorough fine-tuning instructions.

### JAX Dependency

Note this library has some specific requirements for JAX version. Specifically, to use the v1 models (including
 GPT-J 6B), `jax==0.2.12` is required. This in turn depends on `jaxlib==0.1.68`. **If this is not done, you will get
cryptic xmap errors**

However, to use the v2 model code (no publicly released weights), the newest JAX version can be used.
# Citation

To cite this repository:
```
@misc{mesh-transformer-jax,
  author = {Wang, Ben},
  title = {{Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX}},
  howpublished = {\url{https://github.com/kingoflolz/mesh-transformer-jax}},
  year = 2021,
  month = May
}
```

To cite the weights of GPT-J-6B:
```
@misc{gpt-j,
  author = {Wang, Ben and Komatsuzaki, Aran},
  title = {{GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model}},
  howpublished = {\url{https://github.com/kingoflolz/mesh-transformer-jax}},
  year = 2021,
  month = May
}
```

If you use this repository or any of the pretrained weights to do something cool, we would love to hear about it.
Feel free to open a github issue or reach out over email (in profile).

# TODO
- [x] disentangle heads and shards
- [x] test/benchmark on TPU
- [x] implement gradient checkpointing
- [x] fix initialization
- [x] mixed precision
- [x] deal with preemptible TPUs
- [x] test and validate generation
- [x] shard activations instead of replicating for memory efficiency (in v2)
- [x] support ZeRO style sharding (in v2)
