#!/usr/bin/env bash
set -e

# initializes jax and installs ray on cloud TPUs
sudo pip install --upgrade jaxlib jax==0.2.12 ray==1.2.0 fabric dataclasses optax git+https://github.com/deepmind/dm-haiku tqdm cloudpickle smart_open[gcs] einops func_timeout transformers flask