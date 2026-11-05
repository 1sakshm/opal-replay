"""Updater (variant sgd-cosine): the textbook CIFAR recipe a person would try first. No averaging.

SGD + Nesterov with v024's lr, momentum and weight decay, linear warmup, cosine decay to 0 at the last update.
"""

import math

import torch

PEAK_LR = 0.1
RAMP_FRAC = 700 / 42120
MOMENTUM = 0.9
WD = 5e-4


def lr_at(u, total):
    ramp = max(1, round(RAMP_FRAC * total))
    if u <= ramp:
        return PEAK_LR * u / ramp
    return PEAK_LR * 0.5 * (1 + math.cos(math.pi * (u - ramp) / max(1, total - ramp)))


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.total = spec["total_updates"]
        self.opt = torch.optim.SGD([p for _, p in named_params], lr=PEAK_LR, momentum=MOMENTUM, nesterov=True,
                                   weight_decay=WD, foreach=True)

    @property
    def param_groups(self):
        return self.opt.param_groups

    def step(self, grads, update, ctx=None):
        for g in self.opt.param_groups:
            g["lr"] = lr_at(update, self.total)
        self.opt.step()

    def state_dict(self):
        return dict(opt=self.opt.state_dict())

    def load_state_dict(self, state):
        self.opt.load_state_dict(state["opt"])


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
