import argparse
import logging
import math
import os
import omegaconf
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

import data
import models
from augmentations.augmentations import *


logger = get_logger(__name__, log_level="INFO")


def get_arguments():
    example_text = '''
    example:
       accelerate launch --config_file configs/accelerate.yaml train.py --config ./configs/simclr-audioset-vitb32.yaml
    '''
    parser = argparse.ArgumentParser(description="", epilog=example_text)
    parser.add_argument("--config", type=str, default="configs/simclr-audioset-vitb32.yaml")
    args = parser.parse_args()

    return args


def config_to_dict(config):
    config_dict = dict(config)

    def to_dict_saveable(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, omegaconf.listconfig.ListConfig):
            value = list(value)
        return value

    config_dict = {k: to_dict_saveable(v) for k, v in config_dict.items()}

    return config_dict


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main(config):
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        omegaconf.OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))

    # Load scheduler, tokenizer and models.
    model = models.__dict__[config.model.model_name]
    model = model(**config_to_dict(config.model.configuration))

    learning_rate = config.training.learning_rate
    if config.training.scale_lr:
        learning_rate = (
            config.training.learning_rate
            * config.training.gradient_accumulation_steps
            * config.dataset.batch_size
            # * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
        eps=config.training.adam_epsilon,
    )

    # Get the training dataset
    sample_rate = config.dataset.configuration.sample_rate
    transforms = Compose(
        [
            RandomCropExpand(9 * sample_rate, 10 * sample_rate, pad_type="zero"),
            RandomApply(PolarityInversion(), 0.5),
            Noise(min_snr=0.1, max_snr=0.3),
            RandomGain(),
            RandomApply(HighLowPass(sample_rate), p=0.75),
            RandomApply(PitchShift(10 * sample_rate, sample_rate), p=0.5),
            RandomApply(
                RandomBackgroundNoise(
                    noise_root=config.dataset.configuration.dir_noises,
                    sample_rate=sample_rate,
                    segment_size=sample_rate * 10,
                    bank_size=1024,
                    snr_dbs_range=[8, 12],
                ),
                p=0.25,
            ),
            RandomApply(RandomRIR(), p=0.75),
            RandomApply(
                RandomEncoder(sample_rate=sample_rate, codecs=["pcm_mulaw", "g722"]), p=0.5
            ),
        ]
    )
    dataset = data.__dict__[config.dataset.dataset_name]
    dataset = dataset(
        df_path=config.dataset.configuration.df_path,
        transforms=transforms,
        sample_rate=sample_rate,
        num_views=config.dataset.configuration.num_views,
        num_repeat=config.dataset.configuration.num_repeat,
        max_size=config.dataset.configuration.max_size,
    )

    # DataLoaders creation:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=False,
    )

    # Scheduler
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / config.training.gradient_accumulation_steps / accelerator.num_processes
    )
    max_train_steps = num_update_steps_per_epoch * config.training.num_epochs

    lr_scheduler = get_scheduler(
        config.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.lr_warmup_steps
        * config.training.gradient_accumulation_steps,
        num_training_steps=max_train_steps * config.training.gradient_accumulation_steps,
    )

    # Criterion
    criterion = models.NTXentLoss(
        num_views=config.dataset.configuration.num_views, temperature=config.training.temperature
    )

    # Prepare everything with our `accelerator`.
    (model, optimizer, dataloader, lr_scheduler) = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("simclr-training")

    # Train!
    total_batch_size = (
        config.dataset.batch_size
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(" Num examples = {}".format(len(dataset)))
    logger.info(" Num Epochs = {}".format(config.training.num_epochs))
    logger.info(" Instantaneous batch size per device = {}".format(config.dataset.batch_size))
    logger.info(
        " Total train batch size (w. parallel, distributed & accumulation) = {}".format(
            total_batch_size
        )
    )
    logger.info(
        " Gradient Accumulation steps = {}".format(config.training.gradient_accumulation_steps)
    )
    logger.info(" Total optimization steps = {}".format(max_train_steps))
    logger.info(
        " Number of parameters in model: {}".format(sum([l.nelement() for l in model.parameters()]))
    )

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.training.resume_from_checkpoint:
        if config.training.resume_from_checkpoint != "latest":
            path = config.training.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
            path = dirs[-1]
            path = os.path.join(config.output_dir, path)
        accelerator.print("Resuming from checkpoint {}".format(path))
        accelerator.load_state(path)
        global_step = int(path.split("checkpoint-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, config.training.num_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(dataloader):
            # Skip steps until we reach the resumed step
            if (
                config.training.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % config.training.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # regressor step
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                audios = batch["audio"].to(weight_dtype)
                audios = torch.split(audios, split_size_or_sections=1, dim=1)
                audios = torch.cat(audios, dim=0)

                features = model(audios)

                # loss
                logits, labels = criterion(features)
                loss = F.cross_entropy(logits.float(), labels, reduction='mean')

                # accuracy
                top1, top5 = accuracy(logits, labels, topk=(1, 5))

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.dataset.batch_size)).mean()
                train_loss += avg_loss.item() / config.training.gradient_accumulation_steps
                avg_top1 = accelerator.gather(top1.repeat(config.dataset.batch_size)).mean().item()
                avg_top5 = accelerator.gather(top5.repeat(config.dataset.batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {"train_loss": train_loss, "top1": avg_top1, "top5": avg_top5}, step=global_step
                )
                train_loss = 0.0

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "top1": avg_top1,
                "top5": avg_top5,
            }
            progress_bar.set_postfix(**logs)

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            dataloader.dataset.shuffle_class_indexes()
            if (
                epoch + 1
            ) % config.training.save_every_epoch == 0 or epoch == config.training.num_epochs - 1:
                emodel = accelerator.unwrap_model(model)
                save_path = os.path.join(config.output_dir, "checkpoint-{:04d}".format(epoch + 1))
                accelerator.save_state(save_path)
                torch.save(emodel.state_dict(), os.path.join(save_path, "simclr-model.pt"))
                logger.info("Saved state to {}".format(save_path))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        emodel = accelerator.unwrap_model(model)
        torch.save(emodel.state_dict(), os.path.join(config.output_dir, "last_checkpoint.pt"))

    accelerator.end_training()


if __name__ == '__main__':
    args = get_arguments()
    config = omegaconf.OmegaConf.load(args.config)

    main(config)
