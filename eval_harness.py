import argparse
import json

from lm_eval import evaluator, tasks

from mesh_transformer.build_model import build_model
from tasks import EvalHarnessAdaptor


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on.")
    parser.add_argument("--tpu_region", type=str, help="Region of TPU to train on.")
    parser.add_argument("--preemptible", action="store_true")

    parser.add_argument("--config", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    tpu_name = args.tpu
    region = args.tpu_region
    preemptible = args.preemptible

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    tpu_size = params["tpu_size"]
    cores_per_replica = params["cores_per_replica"]

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]
    pe = params["pe"]

    total_batch = per_replica_batch * tpu_size // cores_per_replica * 4

    t = build_model(params, tpu_name, region, preemptible)
    adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")

    step, aux = t.load(bucket, model_dir)
    t.move()

    results = evaluator.evaluate(adaptor, tasks.get_task_dict(["lambada",
                                                               "piqa",
                                                               "hellaswag",
                                                               "winogrande",
                                                               "mathqa",
                                                               "pubmedqa",
                                                               # "boolq",
                                                               # "cb",
                                                               # "copa",
                                                               # "multirc",
                                                               # "record",
                                                               # "wic",
                                                               # "wsc",
                                                               ]), False, 0, None)
    dumped = json.dumps(results, indent=2)
    print(dumped)

    results = evaluator.evaluate(adaptor, tasks.get_task_dict(["lambada_cloze",
                                                               ]), False, 15, None)

    dumped = json.dumps(results, indent=2)
    print(dumped)