#!/usr/bin/env bash
# initializes jax and installs ray on cloud TPUs

# create tempfs for ray shared memory
sudo mkdir /dev/shm -p
sudo mount -t tmpfs -o size=100g tmpfs /dev/shm

sudo pip uninstall -y jax
sudo pip install --upgrade jaxlib==0.1.62 git+https://github.com/google/jax@79040f109ea8ef7d7dc299314f2af742dd174ec3#egg=jax ray fabric dataclasses optax git+https://github.com/deepmind/dm-haiku tqdm cloudpickle smart_open[gcp]