"""Early stopping on a validation metric (RaCA addition; MRA itself has no early stopping).

Works with mmcv 1.4 / mmdet 2.14, where no early-stopping hook exists. The hook runs after the evaluation hook of each
epoch, reads the monitored metric that the evaluation hook wrote to ``runner.log_buffer.output`` (e.g.
``semkitti_SSC_mIoU`` from ``SemanticKittiDataset.evaluate``), and ends training when the metric has not improved by
at least ``min_delta`` for ``patience`` consecutive evaluations. Training stops by shrinking ``runner._max_epochs`` to
the current epoch, which makes ``EpochBasedRunner.run`` leave its loop after the hooks of this epoch have run; the
checkpoint and logger hooks of the epoch therefore still execute. With distributed training the decision is taken on
rank 0 and broadcast, so all ranks stop together.

Config example (``priority`` must lie between the evaluation hook, LOW = 70, and the logger hooks, VERY_LOW = 90,
which clear the log buffer)::

    evaluation = dict(interval=1, save_best='semkitti_SSC_mIoU', rule='greater')
    custom_hooks = [dict(type='EarlyStoppingHook', monitor='semkitti_SSC_mIoU', rule='greater',
                         patience=3, min_delta=0.0, min_epochs=0, priority=80)]

Validation must be on (``tools/train.py`` without ``--no-validate``); otherwise the hook only logs a warning once.

``source='train'`` monitors a training log variable instead, averaged over all iterations of the epoch, e.g. the total
training loss (``monitor='loss'``: the sum of all loss terms, as logged by mmcv; use ``rule='less'``). No validation is
needed for it. Under distributed training mmdet already averages log variables over the ranks. There is no validation
loss: MRA's ``forward_train`` updates the EMA teacher, so it cannot be run on validation frames without side effects.
The record of every evaluation is written to ``<work_dir>/early_stopping.json``.
"""
import json
import os.path as osp

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook, get_dist_info


@HOOKS.register_module()
class EarlyStoppingHook(Hook):

    rules = {'greater': lambda new, best, d: new > best + d,
             'less': lambda new, best, d: new < best - d}

    def __init__(self, monitor='semkitti_SSC_mIoU', rule='greater', patience=3, min_delta=0.0, min_epochs=0,
                 stop_training=True, json_name='early_stopping.json', source='val'):
        assert rule in self.rules, f"rule must be 'greater' or 'less', got {rule!r}"
        assert source in ('val', 'train'), f"source must be 'val' or 'train', got {source!r}"
        self.source, self._sum, self._n = source, 0.0, 0
        assert patience >= 1 and min_delta >= 0
        self.monitor, self.rule, self.patience, self.min_delta = monitor, rule, patience, float(min_delta)
        self.min_epochs, self.stop_training, self.json_name = min_epochs, stop_training, json_name
        self.best, self.best_epoch, self.bad_count, self.history = None, None, 0, []
        self._warned = False

    # ------------------------------------------------------------------ helpers
    def _improved(self, value):
        return self.best is None or self.rules[self.rule](value, self.best, self.min_delta)

    def _decide(self, runner):
        """Returns (value or None, stop) on rank 0; value None means the metric was not found."""
        if self.source == 'train':
            value = self._sum / self._n if self._n else None       # epoch mean of the training log variable
        else:
            value = runner.log_buffer.output.get(self.monitor, None)
        if value is None:
            return None, False
        value = float(value)
        epoch = runner.epoch + 1                                   # 1-based epoch that has just finished
        if self._improved(value):
            self.best, self.best_epoch, self.bad_count = value, epoch, 0
        else:
            self.bad_count += 1
        self.history.append(dict(epoch=epoch, value=value, best=self.best, best_epoch=self.best_epoch,
                                 bad_count=self.bad_count))
        stop = self.bad_count >= self.patience and epoch >= self.min_epochs
        return value, stop

    def _dump(self, runner, stopped):
        rec = dict(monitor=self.monitor, source=self.source, rule=self.rule, patience=self.patience, min_delta=self.min_delta,
                   min_epochs=self.min_epochs, best=self.best, best_epoch=self.best_epoch,
                   stopped_early=stopped, last_epoch=runner.epoch + 1, history=self.history)
        with open(osp.join(runner.work_dir, self.json_name), 'w') as f:
            json.dump(rec, f, indent=1)

    # ------------------------------------------------------------------ hook
    def before_train_epoch(self, runner):
        self._sum, self._n = 0.0, 0

    def after_train_iter(self, runner):
        if self.source == 'train':
            v = (runner.outputs or {}).get('log_vars', {}).get(self.monitor, None)
            if v is not None:
                self._sum += float(v); self._n += 1

    def after_train_epoch(self, runner):
        rank, world_size = get_dist_info()
        value, stop = (None, False)
        if rank == 0:
            value, stop = self._decide(runner)
            if value is None and not self._warned:
                runner.logger.warning(f"EarlyStoppingHook: metric '{self.monitor}' not found after epoch "
                                      f"{runner.epoch + 1}; is validation enabled (train without --no-validate), and does "
                                      f"this hook run after the evaluation hook (priority 70) and before the logger hooks "
                                      f"(priority 90), e.g. priority=80?")
                self._warned = True
            elif value is not None:
                runner.logger.info(f"EarlyStoppingHook: {self.monitor} = {value:.4f} (best {self.best:.4f} at epoch "
                                   f"{self.best_epoch}, {self.bad_count}/{self.patience} epochs without improvement)")
                self._dump(runner, stopped=stop and self.stop_training)
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            flag = torch.tensor([int(stop)], device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
        if stop and self.stop_training:
            if rank == 0:
                runner.logger.info(f"EarlyStoppingHook: stopping after epoch {runner.epoch + 1}; best {self.monitor} = "
                                   f"{self.best:.4f} at epoch {self.best_epoch} (patience {self.patience}).")
            runner._max_epochs = runner.epoch + 1                  # EpochBasedRunner.run exits its while-loop
