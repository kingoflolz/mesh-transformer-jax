#!/usr/bin/env bash
set -e

# initializes jax and installs ray on cloud TPUs
pip install "jax[tpu]>=0.2.18" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo pip install --upgrade ray[default]==1.4.1 fabric dataclasses optax git+https://github.com/deepmind/dm-haiku tqdm cloudpickle smart_open[gcs] einops func_timeout