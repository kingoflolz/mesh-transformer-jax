#!/usr/bin/env bash
set -e

sudo /usr/bin/docker-credential-gcr configure-docker

sudo docker rm libtpu || true
sudo docker create --name libtpu gcr.io/cloud-tpu-v2-images/libtpu:libtpu_20210518_RC00 "/bin/bash" && sudo docker cp libtpu:libtpu.so /lib

# this locks the python executable down to hopefully stop if from being fiddled with...
screen -d -m python -c 'import time; time.sleep(999999999)'

# initializes jax and installs ray on cloud TPUs
sudo pip install --upgrade jaxlib==0.1.67 jax==0.2.12 ray[default]==1.5.1 fabric dataclasses optax git+https://github.com/deepmind/dm-haiku tqdm cloudpickle smart_open[gcs] einops func_timeout