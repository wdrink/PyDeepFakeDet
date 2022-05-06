#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified in 2022 by Fudan Vision and Learning Lab

import copy
import os

import torch

import PyDeepFakeDet.utils.distributed as du
import PyDeepFakeDet.utils.logging as logging

logger = logging.get_logger(__name__)


def make_checkpoint_dir(dir):
    if du.is_master_proc() and not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except Exception:
            pass


def get_checkpoint_dir(path_to_job):
    return os.path.join(path_to_job, "checkpoints")


def get_path_to_checkpoint(path_to_job, epoch, cfg):
    file_name = (
        cfg['MODEL']['MODEL_NAME']
        + '_'
        + cfg['DATASET']['DATASET_NAME']
        + '_'
        + 'epoch_{:05d}'
        + '.pyth'
    )
    file_name = file_name.format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job), file_name)


def is_checkpoint_epoch(cfg, cur_epoch):
    if cur_epoch + 1 == cfg['TRAIN']['MAX_EPOCH']:
        return True
    return (cur_epoch + 1) % cfg['TRAIN']['CHECKPOINT_PERIOD'] == 0


def save_checkpoint(model, optimizer, scheduler, epoch, cfg):
    path_to_job = cfg['TRAIN']['CHECKPOINT_SAVE_PATH']
    if not du.is_master_proc():
        return
    make_checkpoint_dir(get_checkpoint_dir(path_to_job))
    sd = (
        model.module.state_dict() if cfg['NUM_GPUS'] > 1 else model.state_dict()
    )
    normalized_sd = sub_to_normal_bn(sd)

    checkpoint = {
        "epoch": epoch,
        "model_state": normalized_sd,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
        if scheduler is not None
        else None,  # TODO
        "cfg": cfg,
    }

    path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1, cfg)
    with open(path_to_checkpoint, "wb") as f:
        torch.save(checkpoint, f)


def load_checkpoint(
    path_to_checkpoint,
    model,
    data_parallel=True,
    optimizer=None,
    scheduler=None,
    epoch_reset=False,
):
    assert os.path.exists(
        path_to_checkpoint
    ), "Checkpoint '{}' not found".format(path_to_checkpoint)
    logger.info("Loading network weights from {}.".format(path_to_checkpoint))

    ms = model.module if data_parallel else model

    with open(path_to_checkpoint, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    model_state_dict = ms.state_dict()

    checkpoint["model_state"] = normal_to_sub_bn(
        checkpoint["model_state"], model_state_dict
    )

    pre_train_dict = checkpoint["model_state"]
    model_dict = ms.state_dict()
    # Match pre-trained weights that have same shape as current model.
    pre_train_dict_match = {
        k: v
        for k, v in pre_train_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }

    # Weights that do not have match from the pre-trained model.
    not_load_layers = [
        k for k in model_dict.keys() if k not in pre_train_dict_match.keys()
    ]

    # Log weights that are not loaded with the pre-trained weights.
    if not_load_layers:
        for k in not_load_layers:
            logger.info("Network weights {} not loaded.".format(k))

    # Load pre-trained weights.
    ms.load_state_dict(pre_train_dict_match, strict=False)
    epoch = -1

    # Load the optimizer state (commonly not done when fine-tuning)
    if "epoch" in checkpoint.keys() and not epoch_reset:
        epoch = checkpoint["epoch"]
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

    else:
        epoch = -1

    return epoch


def sub_to_normal_bn(sd):
    """
    Convert the Sub-BN paprameters to normal BN parameters in a state dict.
    There are two copies of BN layers in a Sub-BN implementation: `bn.bn` and
    `bn.split_bn`. `bn.split_bn` is used during training and
    "compute_precise_bn". Before saving or evaluation, its stats are copied to
    `bn.bn`. We rename `bn.bn` to `bn` and store it to be consistent with normal
    BN layers.
    Args:
        sd (OrderedDict): a dict of parameters whitch might contain Sub-BN
        parameters.
    Returns:
        new_sd (OrderedDict): a dict with Sub-BN parameters reshaped to
        normal parameters.
    """
    new_sd = copy.deepcopy(sd)
    modifications = [
        ("bn.bn.running_mean", "bn.running_mean"),
        ("bn.bn.running_var", "bn.running_var"),
        ("bn.split_bn.num_batches_tracked", "bn.num_batches_tracked"),
    ]
    to_remove = ["bn.bn.", ".split_bn."]
    for key in sd:
        for before, after in modifications:
            if key.endswith(before):
                new_key = key.split(before)[0] + after
                new_sd[new_key] = new_sd.pop(key)

        for rm in to_remove:
            if rm in key and key in new_sd:
                del new_sd[key]

    for key in new_sd:
        if key.endswith("bn.weight") or key.endswith("bn.bias"):
            if len(new_sd[key].size()) == 4:
                assert all(d == 1 for d in new_sd[key].size()[1:])
                new_sd[key] = new_sd[key][:, 0, 0, 0]

    return new_sd


def c2_normal_to_sub_bn(key, model_keys):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        key (OrderedDict): source dict of parameters.
        mdoel_key (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    if "bn.running_" in key:
        if key in model_keys:
            return key

        new_key = key.replace("bn.running_", "bn.split_bn.running_")
        if new_key in model_keys:
            return new_key
    else:
        return key


def normal_to_sub_bn(checkpoint_sd, model_sd):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        checkpoint_sd (OrderedDict): source dict of parameters.
        model_sd (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    for key in model_sd:
        if key not in checkpoint_sd:
            if "bn.split_bn." in key:
                load_key = key.replace("bn.split_bn.", "bn.")
                bn_key = key.replace("bn.split_bn.", "bn.bn.")
                checkpoint_sd[key] = checkpoint_sd.pop(load_key)
                checkpoint_sd[bn_key] = checkpoint_sd[key]

    for key in model_sd:
        if key in checkpoint_sd:
            model_blob_shape = model_sd[key].shape
            c2_blob_shape = checkpoint_sd[key].shape

            if (
                len(model_blob_shape) == 1
                and len(c2_blob_shape) == 1
                and model_blob_shape[0] > c2_blob_shape[0]
                and model_blob_shape[0] % c2_blob_shape[0] == 0
            ):
                before_shape = checkpoint_sd[key].shape
                checkpoint_sd[key] = torch.cat(
                    [checkpoint_sd[key]]
                    * (model_blob_shape[0] // c2_blob_shape[0])
                )
                logger.info(
                    "{} {} -> {}".format(
                        key, before_shape, checkpoint_sd[key].shape
                    )
                )
    return checkpoint_sd


def load_test_checkpoint(cfg, model):
    assert (
        cfg['TEST']['TEST_CHECKPOINT_PATH'] != ''
    ), 'TEST_CHECKPOINT_PATH is empty!'
    load_checkpoint(
        cfg['TEST']['TEST_CHECKPOINT_PATH'],
        model,
        cfg['NUM_GPUS'] > 1,
    )


def load_train_checkpoint(model, optimizer, scheduler, cfg):
    if cfg['TRAIN']['CHECKPOINT_LOAD_PATH'] != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = load_checkpoint(
            cfg['TRAIN']['CHECKPOINT_LOAD_PATH'],
            model,
            cfg['NUM_GPUS'] > 1,
            optimizer,
            scheduler,
            epoch_reset=cfg['TRAIN']['CHECKPOINT_EPOCH_RESET'],
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    return start_epoch
