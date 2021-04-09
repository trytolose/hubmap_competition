from warmup_scheduler import GradualWarmupScheduler
import torch


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(
        self, optimizer, multiplier, total_epoch, cosine_epo, after_scheduler=None
    ):
        self.after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cosine_epo
        )
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, self.after_scheduler,
        )

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
