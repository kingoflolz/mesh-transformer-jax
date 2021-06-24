#!/usr/bin/env bash
set -e

sudo /usr/bin/docker-credential-gcr configure-docker

sudo docker rm libtpu || true
sudo docker create --name libtpu gcr.io/cloud-tpu-v2-images/libtpu:libtpu_20210518_RC00 "/bin/bash" && sudo docker cp libtpu:libtpu.so /lib

# initializes jax and installs ray on cloud TPUs
sudo pip install --upgrade jaxlib jax==0.2.12 ray==1.2.0 fabric dataclasses optax==0.0.6 git+https://github.com/deepmind/dm-haiku tqdm cloudpickle smart_open[gcs] einops func_timeout