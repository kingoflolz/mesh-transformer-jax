#!/usr/bin/env bash
set -e

# initializes jax and installs ray on cloud TPUs
sudo pip install "jax[tpu]==0.2.14" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo pip install --upgrade ray==1.2.0 fabric dataclasses optax==0.0.6 git+https://github.com/deepmind/dm-haiku tqdm cloudpickle smart_open[gcs] einops func_timeout