
import os
import time
import jax
import optax
import threading
import numpy as np
import logging
from queue import Queue, Empty
from jax.experimental import maps
from transformers import GPT2TokenizerFast

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

logger = logging.getLogger(__name__)

# This prevents fastapi from creating multiple models in multi-worker mode
# leading to OOM crashes
gptj_model = None
gptj_model_lock = threading.Lock()

def compile_model():
    global gptj_model
    with gptj_model_lock:
        if gptj_model:
            return
        gptj_model = GPTJ()

def get_gptj_model():
    compile_model()
    return gptj_model

def timer(start_time=None):
    if not start_time:
        return time.time()
    return time.time() - start_time


class GPTJ:
    def __init__(self):
        self.params = {
            "layers": 28,
            "d_model": 4096,
            "n_heads": 16,
            "n_vocab": 50400,
            "norm": "layernorm",
            "pe": "rotary",
            "pe_rotary_dims": 64,
            "seq": 2048,
            "cores_per_replica": 8,
            "per_replica_batch": 1,
            "sampler": nucleaus_sample,
            "optimizer": optax.scale(0)
        }
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.queue_ids = {}
        self.qidx = 0
        self.queue = Queue()
        self.network = None
        self.lock = threading.Lock()
        self._alive_time = timer()
    
    def load_model(self):
        if self.network:
            logger.info('Attempting to reload model when model is loaded. Returning')
            return
        with self.lock:
            logger.info('Loading Model')
            start = timer()
            logger.info(f"JAX Devices: {jax.device_count()}")
            logger.info(f"JAX Runtime Initialized in {timer(start):.06} secs")
            mesh_shape = (jax.device_count() // self.params['cores_per_replica'], self.params['cores_per_replica'])
            self.devices = np.array(jax.devices()).reshape(mesh_shape)
            self.total_batch = self.params['per_replica_batch'] * jax.device_count() // self.params['cores_per_replica'] * 8
            maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(self.devices, ('dp', 'mp')))
            network = CausalTransformer(self.params)
            logger.info(f'Loading Checkpoint')
            network.state = read_ckpt(network.state, "/app/model/", self.devices.shape[1])
            logger.info(f"GPTJ Network loaded in {timer(start):.06} secs. Total Batch Size: {self.total_batch}")
            del network.state["opt_state"]
            network.state = network.move_xmap(network.state, np.zeros(self.params['cores_per_replica']))
            self.network = network

    
    def start_background(self):
        with self.lock:
            t = threading.Thread(target=self.background)
            t.start()
    
    def prepare_item(self, context, length=256):
        tokens = self.tokenizer.encode(context)
        logger.info(tokens)
        token_length = len(tokens)
        pad_amount = self.params['seq'] - token_length
        pad_amount = max(pad_amount, 0)
        padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)[-self.params['seq']:]
        return {'tokens': padded_tokens, 'length': token_length}
    
    # Single Item - Not Tested
    def infer(self, context, top_p=0.9, top_k=40, temp=1.0, length=256, **kwargs):
        item = self.prepare_item(context, length)
        batched_tokens = np.array([item['tokens']] * self.total_batch)
        batched_lengths = np.array([item['length']] * self.total_batch)
        start = timer()
        output = self.network.generate(
            batched_tokens, batched_lengths, length, 
            {
                "top_p": np.ones(self.total_batch) * top_p,
                "top_k": np.ones(self.total_batch) * top_k,
                "temp": np.ones(self.total_batch) * temp,
            }
        )
        samples = []
        decoded_tokens = output[1][0]
        end_time = timer(start)
        for o in decoded_tokens[:, :, 0]:
            res = {
                'context': context,
                'completion': self.tokenizer.decode(o),
                'time': end_time
            }
            samples.append(res)
        logger.info(f"Completion done in {end_time:06} secs")
        return samples
    
    def infer_batch(self, batch, **kwargs):
        logger.info(f'Starting Inference on Batch')
        batch_items = {'tokens': [], 'lengths': [], 'top_p': [], 'top_k': [], 'temp': []}
        max_lengths, contexts = [], []
        for req in batch:
            req = self.to_data(req)
            item = self.prepare_item(req['context'], req['length'])
            batch_items['tokens'].append(item['tokens'])
            batch_items['lengths'].append(item['length'])
            batch_items['top_p'].append(req['top_p'])
            batch_items['top_k'].append(req['top_k'])
            batch_items['temp'].append(req['temp'])
            max_lengths.append(req['length'])
            contexts.append(req['context'])
        
        max_length = max(max_lengths)
        for key, vals in batch_items.items():
            batch_items[key] = np.array(vals)
        start = timer()
        logger.info(f'Completed Preparing Batch')
        output = self.network.generate(
            batch_items['tokens'], batch_items['lengths'], max_length, 
            {
                "top_p": batch_items['top_p'],
                "top_k": batch_items['top_k'],
                "temp": batch_items['temp'],
            }
        )
        logger.info(f'Completed Generation')
        samples = []
        end_time = timer(start)
        for pred, ctx in zip(output[1][0][:, :, 0], contexts):
            res = {
                'context': ctx,
                'completion': self.tokenizer.decode(pred),
                'time': end_time
            }
            samples.append(res)
        logger.info(f"Completion done in {end_time:06} secs")
        return samples

    def add_to_queue(self, item):
        self.qidx += 1
        self.queue.put({'item': self.to_data(item), 'qidx': self.qidx})
        self.queue_ids[self.qidx] = Queue()
        return {'qid': self.qidx}

    def wait_for_queue(self, qid):
        if not self.queue_ids.get(qid):
            return {'Error': 'QID not found'}
        return self.queue_ids[qid].get()

    def background(self):
        logger.info(f'Init Background')
        maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(self.devices, ('dp', 'mp')))
        while True:
            batch, qids = [], []
            while len(batch) <= self.total_batch:
                try:
                    req = self.queue.get(block=False)
                    logger.info(f'Got Queue Item: {req}')
                    batch.append(req['item'])
                    qids.append(req['qidx'])
                    
                except Empty:
                    if len(batch):
                        break
                    else:
                        time.sleep(0.01)
            batch_size = len(batch)
            logger.info(f'Working on Batch: {batch_size} - {qids}')
            while len(batch) < self.total_batch:
                batch.append(self.placeholder_item)
            start = timer()
            results = self.infer_batch(batch)
            for res, qid in zip(results, qids):
                self.queue_ids[qid].put(res)
            logger.info(f'Completed Current Batch of {batch_size} Items in {timer(start):.2f} secs')

    @property
    def placeholder_item(self):
        return {'context': 'nada', 'top_p': 0.9, 'top_k': 40, 'temp': 1.0, 'length': 1}
    
    def to_data(self, item):
        try:
            return {'context': item.context, 'top_p': item.top_p, 'top_k': item.top_k, 'temp': item.temp, 'length': item.length}
        except:
            return {'context': item.get('context', ''), 'top_p': item.get('top_p', 0.9), 'top_k': item.get('top_k', 40), 'temp': item.get('temp', 1.0), 'length': item.get('length', 256)}

    @property
    def alive_time(self):
        return timer(self._alive_time)