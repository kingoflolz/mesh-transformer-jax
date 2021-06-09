#!/usr/bin/env bash
set -e

rm -r mesh-transformer-jax || true

git clone https://github.com/kingoflolz/mesh-transformer-jax
pip install -r mesh-transformer-jax/requirements.txt
pip install mesh-transformer-jax/ jax==0.2.12

pushd mesh-transformer-jax || exit
screen -d -m python3 device_serve.py --config configs/6B_roto_256.json