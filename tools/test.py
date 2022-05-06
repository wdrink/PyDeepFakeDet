import numpy as np
import torch
import torch.nn.functional as F
from timm.utils import accuracy

import PyDeepFakeDet.utils.checkpoint as cu
import PyDeepFakeDet.utils.distributed as du
import PyDeepFakeDet.utils.logging as logging
from PyDeepFakeDet.utils.build_helper import (
    build_dataloader,
    build_dataset,
    build_loss_fun,
    build_model,
)
from PyDeepFakeDet.utils.meters import AucMetric, MetricLogger

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(
    test_loader, model, cfg, cur_epoch=None, writer=None, mode='Test'
):
    criterion = build_loss_fun(cfg)

    metric_logger = MetricLogger(delimiter="  ")
    auc_metrics = AucMetric(cfg['NUM_GPUS'])
    header = mode + ':'

    model.eval()

    for samples in metric_logger.log_every(test_loader, 10, header):
        samples = dict(
            zip(
                samples,
                map(
                    lambda sample: sample.cuda(non_blocking=True),
                    samples.values(),
                ),
            )
        )
        with torch.cuda.amp.autocast(enabled=cfg['AMP_ENABLE']):
            outputs = model(samples)
            loss = criterion(outputs, samples)
            preds = F.softmax(outputs['logits'], dim=1)[:, 1]
            auc_metrics.update(samples['bin_label'].squeeze(dim=1), preds)
        acc1 = accuracy(
            outputs['logits'], samples['bin_label'], topk=(1,)
        )
        batch_size = samples['img'].shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1[0].item(), n=batch_size)

    auc_metrics.synchronize_between_processes()
    metric_logger.synchronize_between_processes()
    if writer and cur_epoch is not None:
        writer.add_scalar(
            'acc1', metric_logger.acc1.global_avg, global_step=cur_epoch
        )
        writer.add_scalar('auc', auc_metrics.auc, global_step=cur_epoch)

    logger.info(
        '* Acc@1 {top1.global_avg:.3f} Auc {auc:.3f} loss {losses.global_avg:.3f}'.format(
            top1=metric_logger.acc1,
            auc=auc_metrics.auc,
            losses=metric_logger.loss,
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test(local_rank, num_proc, init_method, shard_id, num_shards, cfg):
    du.init_process_group(
        local_rank=local_rank,
        local_world_size=num_proc,
        shard_id=shard_id,
        num_shards=num_shards,
        init_method=init_method,
    )
    np.random.seed(cfg['RNG_SEED'])
    torch.manual_seed(cfg['RNG_SEED'])

    logging.setup_logging(cfg, mode='test')

    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)

    test_dataset = build_dataset('test', cfg)
    test_loader = build_dataloader(test_dataset, 'test', cfg)

    logger.info("Testing model for {} iterations".format(len(test_loader)))

    test_stats = perform_test(test_loader, model, cfg)
    logger.info(
        f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.1f}%"
    )
