#!/usr/bin/env bash
set -e

# initializes jax and installs ray on cloud TPUs
sudo pip install --upgrade jaxlib==0.1.64 jax==0.2.11 ray fabric dataclasses optax git+https://github.com/deepmind/dm-haiku tqdm cloudpickle smart_open[gcp]