import os
import sys
import requests


def get_metadata(key):
  return requests.get(
      'http://metadata.google.internal/computeMetadata'
      '/v1/instance/attributes/{}'.format(key),
      headers={
          'Metadata-Flavor': 'Google'
      }).text


worker_id = get_metadata('agent-worker-number')
accelerator_type = get_metadata('accelerator-type')
worker_network_endpoints = get_metadata('worker-network-endpoints')

os.environ['CLOUD_TPU_TASK_ID'] = worker_id
os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'

accelerator_type_to_host_bounds = {
    'v2-8': '1,1,1',
    'v2-32': '2,2,1',
    'v2-128': '4,4,1',
    'v2-256': '4,8,1',
    'v2-512': '8,8,1',
    'v3-8': '1,1,1',
    'v3-32': '2,2,1',
    'v3-128': '4,4,1',
    'v3-256': '4,8,1',
    'v3-512': '8,8,1',
    'v3-1024': '8,16,1',
    'v3-2048': '16,16,1',
}

if not accelerator_type.endswith("-8"):
    os.environ['TPU_HOST_BOUNDS'] = accelerator_type_to_host_bounds[accelerator_type]
    os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = worker_network_endpoints.split(',')[0].split(':')[2] + ':8476'
    os.environ['TPU_MESH_CONTROLLER_PORT'] = '8476'

os.system(f"ray start --address={sys.argv[1]} --resources='" + '{"tpu": 1}\'')
