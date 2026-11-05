"""Updater (variant adamw-wsd): the training script's own AdamW, with the warmup-hold-decay schedule of the agent's
final program (v014) in place of the script's warmup-then-constant one. No Muon, no per-layer multipliers."""

import torch

BASE_LR = 1e-3
WARMUP_END = 200
DECAY_START_FRAC = 2000 / 6000
DECAY_END_FRAC = 0.95
FINAL_FACTOR = 0.03


def lr_factor(update, total):
    w = min(1.0, update / WARMUP_END)
    s, e = DECAY_START_FRAC * total, DECAY_END_FRAC * total
    if update <= s:
        return w
    x = min(1.0, (update - s) / max(1.0, e - s))
    return w * (1.0 - x * (1.0 - FINAL_FACTOR))


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.opts = spec["user_optimizers"]
        self.total = spec["total_updates"]

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self, grads, update, ctx=None):
        f = lr_factor(update, self.total)
        for g in self.param_groups:
            g["lr"] = BASE_LR * f
        for o in self.opts:
            o.step()

    def state_dict(self):
        return dict(opts=[o.state_dict() for o in self.opts])

    def load_state_dict(self, state):
        for o, s in zip(self.opts, state["opts"]):
            o.load_state_dict(s)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
