
import torch
import torch.nn as nn


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size, warmup_steps, factor=1.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
        return [base_lr * scale for base_lr in self.base_lrs]