import argparse
import json
import time

import numpy as np
import wandb
from tqdm import tqdm

from mesh_transformer.build_model import build_model
from lm_eval import evaluator, tasks
from tasks.eval_harness import EvalHarnessAdaptor
from tfrecord_loader import TFRecordNewInputs
import multiprocessing


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on.")
    parser.add_argument("--tpu_region", type=str, help="Region of TPU to train on.")
    parser.add_argument("--preemptible", action="store_true")

    parser.add_argument("--config", type=str, default=None, help="Config file location")

    parser.add_argument("--new", action="store_true", help="If set, deletes previous checkpoint, if it exists, and "
                                                           "starts a new training run")

    parser.add_argument("--version", type=int, default=1, help="Choose which model version to use")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # huggingface tokenizers gets very angry if you fork
    multiprocessing.set_start_method("spawn")

    args = parse_args()
    params = json.load(open(args.config))

    if args.new:
        print(f"Starting experiment {params['name']} from scratch! "
              f"all data in gs://{params['bucket']}/{params['model_dir']}/ will be deleted")
        input("Hit enter to continue")

    tpu_name = args.tpu
    region = args.tpu_region
    preemptible = args.preemptible
    clean_start = args.new

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

    val_batches = params["val_batches"]
    val_every = params["val_every"]
    ckpt_every = params["ckpt_every"]
    keep_every = params["keep_every"]
    eval_tasks = params["eval_harness_tasks"]
    total_steps = params["total_steps"]

    pe = params["pe"]
    assert pe in ["fixed", "rotary", "t5"]

    t = build_model(params, tpu_name, region, preemptible, version=args.version)

    try:
        t.save(0, bucket, model_dir, init=True, overwrite=clean_start)
        step = 0
        train_load_restore = None
    except Exception as e:
        print(f"Save failed with error {e}, trying to load instead...", e)
        step, aux = t.load(bucket, model_dir)
        train_load_restore = aux.get("train_loader", None)

        if train_load_restore is None:
            print("Failed to restore train loader state")

    train_dataset = TFRecordNewInputs(f"data/{params['train_set']}",
                                      batch_size=(
                                          gradient_accumulation_steps,
                                          per_replica_batch * tpu_size // cores_per_replica),
                                      sample_size=params['seq'],
                                      restore_state=train_load_restore)

    global_val_batch = int(per_replica_batch * tpu_size // cores_per_replica * params.get("val_batch_multiplier", 1))

    val_sets = {}

    for k, v in params['val_set'].items():
        val_sets[k] = TFRecordNewInputs(f"data/{v}",
                                        batch_size=(global_val_batch,),
                                        sample_size=seq)

    # use dynamic seq length unless pe is fixed
    adaptor = EvalHarnessAdaptor(t,
                                 seq,
                                 global_val_batch,
                                 shrink=pe != "fixed",
                                 min_seq=1024 if args.version == 2 else None)  # work around suboptimal pjit layout

    start = time.time()
    t.train(train_dataset.get_samples())
    print(f"Train fn compiled in {time.time() - start:.06}s")

    start = time.time()
    for val_set in val_sets.values():
        t.eval(val_set.get_samples())
    print(f"Eval fn compiled in {time.time() - start:.06}s")

    project = params.get("wandb_project", "mesh-transformer-jax")
    wandb.init(project=project, entity="eleutherai", name=params["name"], config=params)

    eval_task_dict = tasks.get_task_dict(eval_tasks)

    pbar = tqdm(initial=step, total=total_steps, desc="Training progress")

    while True:
        loss, last_loss = t.train(train_dataset.get_samples())
        wandb.log({'train/loss': loss, 'train/last_loss': last_loss}, step)

        if (step % ckpt_every == 0 and step) or step == total_steps:
            t.save(step, bucket, model_dir,
                   aux={"train_loader": train_dataset.get_state()},
                   init=False,
                   delete_old=step % keep_every != 0)

            if step == total_steps:
                print("training completed!")
                exit()

        if step % val_every == 0:
            for name, val_set in val_sets.items():
                val_loss = []
                for i, _ in tqdm(zip(val_set.sample_once(), range(val_batches)),
                                 desc=f"validation for step {step}, set {name}",
                                 total=val_batches):
                    val_loss.append(t.eval(i))
                val_loss = np.array(val_loss).mean()
                print(f"validation loss for step {step}, set {name}: {val_loss}")

                wandb.log({f'val/loss_{name}': float(val_loss)}, step)

            results = evaluator.evaluate(adaptor, eval_task_dict, False, 0, None)

            flat_results = {}

            for task_name, task_res in results["results"].items():
                version = results["versions"][task_name]
                for metric_name, metric_res in task_res.items():
                    flat_results[f"{task_name}-v{version}/{metric_name}"] = float(metric_res)

            dumped = json.dumps(results, indent=2)
            print(f"step {step} val results: {dumped}")
            wandb.log(flat_results, step)
        step += 1

        pbar.set_postfix({'loss': loss, 'last_loss': last_loss})
        pbar.update()
