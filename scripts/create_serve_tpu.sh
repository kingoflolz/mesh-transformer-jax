gcloud alpha compute tpus tpu-vm create "$1" --zone europe-west4-a --accelerator-type v3-8 --version v2-alpha
sleep 120
gcloud alpha compute tpus tpu-vm ssh "$1" --zone europe-west4-a --command connor@sparse:~/kindiana/mesh-transformer-jax --command="$(< scripts/deploy_server.sh)"