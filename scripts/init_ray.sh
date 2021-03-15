#!/usr/bin/env bash
# initializes jax and installs ray on cloud TPUs

# create tempfs for ray shared memory
sudo mkdir /dev/shm -p
sudo mount -t tmpfs -o size=100g tmpfs /dev/shm

sudo pip install --upgrade jaxlib==0.1.62
sudo pip install --upgrade jax ray fabric dataclasses optax git+https://github.com/deepmind/dm-haiku