#!/usr/bin/env bash
set -e

# initializes jax and installs ray on cloud TPUs
sudo pip install --upgrade jaxlib jax ray fabric dataclasses optax git+https://github.com/deepmind/dm-haiku tqdm cloudpickle smart_open[gcs] einops