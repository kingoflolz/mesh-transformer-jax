# How to Fine-Tune GPT-J - The Basics

Before anything else, you'll likely want to apply for access to the TPU Research Cloud (TRC). Combined with a Google Cloud free trial, that should allow you to do everything here for free. Once you're in TRC, you need to create a project, then with the name of the new project fill out the form that was emailed to you. Use the script `create_finetune_tfrecords.py` to prepare your data as tfrecords; I might do a separate guide on that. Another thing you might want to do is fork the mesh-transformer-jax repo to make it easier to add and modify the config files.

0. [Install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install). We'll need it later.

1. If you didn't make a project and activate TPU access through TRC yet (or if you plan on paying out of pocket), [make one now](https://console.cloud.google.com/projectcreate).

2. TPUs use Google Cloud buckets for storage, go ahead and [create one now](https://console.cloud.google.com/storage/create-bucket). Make sure it's in the region the TPU VM will be; the email from TRC will tell you which region(s) you can use free TPUs in.

3. You'll need the full pretrained weights in order to fine-tune the model. [Download those here](https://the-eye.eu/public/AI/GPT-J-6B/step_383500.tar.zstd).

Now that you have a bucket on the cloud and the weights on your PC, you need to upload the weights to the bucket in two steps:

4. Decompress and extract `GPT-J-6B/step_383500.tar.zstd` so you're left with the uncompressed folder containing the sharded checkpoint.

5. Open the Google Cloud SDK and run the following command, replacing the path names as appropriate: `gsutil -m cp -R LOCAL_PATH_TO/step_383500 gs://YOUR-BUCKET`. If that works, the console will show the files being uploaded. *Note: Took about 12 hours for me, uploading to the Netherlands from California; hopefully you'll have a better geographic situation than I did! I also initially made the mistake of uploading the still-packed .tar. Don't do that, TPU VMs don't have enough local storage for you to unpack it. To avoid needing to re-upload, I had to unpack it in Colab.*

You'll want to upload tfrecords of your data as well, you can do that here or through the web interface, but trust me when I say you don't want to upload the nearly 70GB weights through the web interface.

Note that steps 6 and 7, preparing the index and config files, can be done later on by editing the base repo in the VM's text editor. It's more efficient to instead make these changes to your own fork of the repo as follows:

6. In the data folder, create a new file `foo.train.index`, replace foo with whatever you want to refer to your dataset as. For each tfrecord in your bucket that you intend to train with, add the path as a line in the index. Make `foo.val.index` and do the same for your validation dataset (if you have one). See the existing files for examples.

7. Duplicate the config file `6B_roto_256.json`, rename it to something appropriate for your project. Open it up and make these edits:
   - `tpu_size`: Change from `256` to `8`
   - `bucket`: Change to your bucket
   - `model_dir`: Change to the directory you'd like to save your checkpoints in
   - `train_set` and `val_set`: Change to the index files from the last step
   - `eval_harness_tasks`: Can be removed if you don't plan on using the eval harness
   -  `val_every` & `ckpt_every` & `keep_every`: Usage should be intuitive. Don't set the `foo_every` values to 0 though or you'll get a divide by zero error. If you don't have a `val_set`, just set `val_every` to something higher than `total_steps`.
   - `val_batches`: This should equal the number of sequences in your val dataset. You can find this number at the end of the .tfrecords file produced by `create_finetune_tfrecords.py`
   - `name`: Change to a name for your model.
   - `warmup_steps`, `lr`, `val_batches`, etc.: see the *Learning Rate Notes* section at the end of the guide.


8. Push the changes to your GitHub repo.

9. Follow [this guide](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm) up to and including the step **"Connect to your Cloud TPU VM"**.

At this point you should have remote access to the TPU VM!

10. In the new VM terminal, type `git clone https://github.com/kingoflolz/mesh-transformer-jax` (or, preferably, your own fork, after pushing the config and index files)

11. Move to the new directory with `cd mesh-transformer-jax` and run `pip install -r requirements.txt`. Since the requirements.txt file doesn't pin the exact jax version required for finetuning, run `pip install jax==0.2.12` and you'll be all set.

12. Finally, run `python3 device_train.py --config=YOUR_CONFIG.json --tune-model-path=gs://YOUR-BUCKET/step_383500/`. If everything is set up correctly this will begin the fine-tuning process. First the model has to be loaded into memory; when `loading network` displayed on the console it took about 10-15 minutes before the next step, setting up WandB for logging. Option 3 allows you to skip that if you aren't using WandB. A step 1 checkpoint will save, and the real training will start. If you have a small dataset, this will go by quickly; TPU VMs can train at a rate of ~5000 tokens/second.

13. You did it! Now don't forget any clean up steps you need to take like shutting down your TPU VM or removing unneeded data in buckets, so that you don't have any unexpected charges from Google later.

## Now what?

This guide is labeled "The Basics", anything we haven't covered so far is out of scope, but go check out the rest of the repository! Try `python3 device_sample.py --config=configs/YOUR_CONFIG.json` for a basic sampling interface. Use `slim_model.py` to prepare an easier-to-deploy slim version of your new weights for inference. Experiment!

### Running with HuggingFace
To use the model in HuggingFace's `transformer` library using pytorch, you'll need to transfer the weights
into a format that it recognizes. This can be done using `to_hf_weights.py`. It's recommended that you use `slim_model.py` before attempting to move the weights to a pytorch/transformer format. Use `python to_hf_weights.py --help` to see usage details.

*note: as of 9/1/2021, GPT-J has been merged into the `main` branch of `transformers` but has not yet been put into production. Run `pip install git+https://github.com/huggingface/transformers#transformers` to install the current `main` branch.

## Learning Rate Notes

**Thanks to nostalgebraist for talking about this!** They're the one who explained this part on Discord, I'm just paraphrasing really:

The first thing you want to determine is how long a training epoch will be. `gradient_accumulation_steps` is your batch size, it defaults to `16`, nostalgebraist recommends `32`. Your .tfrecord files should have a number in the file name indicating how many sequences are in the dataset. Divide that number by the batch size and the result is how many steps are in an epoch. Now we can write the schedule.

`lr` is recommended to be between `1e-5` and `5e-5`, with `end_lr` set to 1/5 or 1/10 of `lr`.
`weight_decay` can remain `0.1`. `total_steps` should be at least one epoch, possibly longer if you have a validation
set to determine your training loss with.
`warmup_steps` should be 5-10% of total, and finally `anneal_steps` should be `total_steps - warmup_steps`.
(The `lr` is set to `end_lr` after `warmup_steps+anneal_steps` and then keeps training until `total_steps`,
but usually you should stop after annealing is done)

To illustrate: I have a small dataset that tokenized into 1147 sequences as a .tfrecord. Dividing by `gradient_accumulation_steps` set to `16`, rounding up to ensure I use all the data, equals 72 steps per epoch. I'll set `lr` to `5e-5`, `end_lr` to a fifth of that, `1e-5`; that may be too much, it's on the high end of the recommended range. I'll set `total_steps` to `72` for one epoch, since I don't have a validation set. Then I'll set `anneal_steps` to `65` and `warmup_steps` to `7`. Simple as that, but you may need to fiddle with the specifics on your own.
