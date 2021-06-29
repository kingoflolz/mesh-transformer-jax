# Deploying GPT-J (TPU VM) in a Docker Container

This PR enables you to run GPT-J for Inference in a Docker Container, and has run stable over the past week. 

The function currently requires 2 calls, the first is a POST call to /model/predict with the params

```
context: str
temp: Optional[float] = 1.0
top_p: Optional[float] = 0.9
top_k: Optional[int] = 50
length: Optional[int] = 256
```

This returns an ID (int) that is used to retrieve the prediction once the prediction is complete, rather than waiting until the prediction is done.

The second function call is a POST call to /model/get_prediction with `qid=id` that was returned.

## Getting Started

You will need the following:
1) TPU VM: A TPU (v2-8 works fine)
2) Proxy VM: a second VM within the same zone. This VM will only serve to proxy all requests to the TPUVM, so it does not require a lot of resources.
3) [Docker + Docker Compose](https://docs.docker.com/compose/install/) Installed on both VMs
4) TPU VM and Proxy VM on the same GCP Network (default for most)
5) Ports 80 and 443 exposed on the Proxy VM
6) Your DNS records pointing to the External IP for Proxy VM.

The GPTJ Checkpoint should be downloaded locally and the volume should be set as the variable MODEL_DIR in the .env file. This mounts the volume to the container, preventing multiple re-downloads.

## Files to Modify

`.env`

```bash
# Modify these values in .env
DRYRUN=false # Set to True to prevent SSL Lets Encrypt Domain validation when testing
DOMAIN=yourdomain.com # Set to your domain 
SUBDOMAIN=gptj.api # Set to your subdomain. The endpoint will be gptj.api.yourdomain.com
NGINX_PATH=./nginx_proxyvm.conf # Modify this conf file with the Internal IP of the TPU VM. This value will not change.
CERTS_PATH=/path/to/certs/ # Set this to a path on the Proxy VM that be used to store SSL Certs. This path will automatically be created.
DNS_KEYS=/path/to/keys/ # Set this to a path on the Proxy VM that be used to store DNS Keys. This path will automatically be created.
MODEL_DIR=/your/path/to/model/step_383500 # Set this to the path on TPU VM that the model is stored in
TPU_NAME=YOUR_TPU_NAME # Set this to your TPU Name as created
```

`nginx_proxyvm.conf`

Modify the IP to your TPU VM's *Internal* IP Address


## Running the Docker Container

Assuming you are in this directory as the working directory.

On TPU VM:

`docker-compose up -d`

On Proxy VM: 

`docker-compose -f compose-proxy.yaml up`

## Final Notes

The reason you can't directly serve from the TPU VM is because there is no control over the TPU VM's port firewalls. I go into more details about [Dockerization within TPU VMs here](https://trisongz.medium.com/accessing-your-tpus-in-docker-containers-with-tpu-vm-e944f5909dd4). 

The API is _very_ barebones and stripped down for simplicity. It's meant as a starting point and can be further extended for your needs.

