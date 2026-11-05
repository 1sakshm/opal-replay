"""Updater (variant swa-only): only where the agent's final program (v024) ended up, with none of the path to it.

SGD + Nesterov (v024's lr, momentum, weight decay, per-role multipliers), lr held at 0.1 from the start, then v024's six
late cycles (0.1 / 0.15). One EMA per cycle over its last 1000 updates. At FINAL_AT the weights become the mean of the
six EMAs and lr goes to 0, so BatchNorm stats settle on them. No snapshots of earlier phases, no restores, no history.
Update numbers are v024's; they scale with the run length so a short run walks the same stages.
"""

import torch

REF_TOTAL = 42120
RAMP = 700
HOLD_LR = 0.1
MOMENTUM = 0.9
WD = 5e-4
ROLE_MULT = {"conv": 1.0, "bn": 0.5, "fc": 1.0}
CYCLES = [(31300, 32700, 0.1), (32700, 34100, 0.15), (34600, 36000, 0.1),
          (36000, 37400, 0.15), (37400, 38800, 0.1), (38800, 40200, 0.15)]
CYCLE_RAMP = 200
EMA_LEN = 1000
EMA_DECAY = 0.998
FINAL_AT = 40200


def role_of(name):
    if name.startswith("fc."):
        return "fc"
    if name.endswith(".weight") and name.count(".") >= 1 and ("conv" in name or name in ("stem.0.weight",) or "shortcut.0" in name):
        return "conv"
    return "bn"


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.by_name = dict(named_params)
        self.total = spec["total_updates"]
        self.scale = self.total / REF_TOTAL
        self.cycles = [(round(s * self.scale), round(e * self.scale), lr) for s, e, lr in CYCLES]
        self.final_at = round(FINAL_AT * self.scale)
        self.ramp = max(1, round(RAMP * self.scale))
        self.cycle_ramp = max(1, round(CYCLE_RAMP * self.scale))
        self.ema_len = round(EMA_LEN * self.scale)
        self.decay = 1.0 - (1.0 - EMA_DECAY) / self.scale  # the same window in updates after scaling
        groups = {}
        for n, p in named_params:
            groups.setdefault(role_of(n), []).append(p)
        self.opt = torch.optim.SGD(
            [dict(params=ps, lr=HOLD_LR, role=r) for r, ps in groups.items()],
            lr=HOLD_LR, momentum=MOMENTUM, nesterov=True, weight_decay=WD, foreach=True,
        )
        self.ema, self.ema_tag, self.avgs = {}, None, []

    @property
    def param_groups(self):
        return self.opt.param_groups

    def base_lr(self, u):
        if u >= self.final_at:
            return 0.0
        prev = HOLD_LR
        for s, e, lr in self.cycles:
            if s <= u < e:
                return prev + (lr - prev) * min(1.0, (u - s) / self.cycle_ramp)
            prev = lr if u >= e else prev
        return HOLD_LR * min(1.0, u / self.ramp)

    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        if update == self.final_at and self.avgs:
            for n, p in self.by_name.items():
                p.copy_(torch.stack([a[n] for a in self.avgs]).mean(0))
            self.opt.state.clear()
            return
        lr = self.base_lr(update)
        for g in self.opt.param_groups:
            g["lr"] = lr * ROLE_MULT[g["role"]]
        self.opt.step()
        for s, e, _ in self.cycles:
            if e - self.ema_len <= update < e:
                if self.ema_tag != s:
                    self.ema, self.ema_tag = {n: p.detach().clone() for n, p in self.by_name.items()}, s
                else:
                    for n, p in self.by_name.items():
                        self.ema[n].lerp_(p, 1.0 - self.decay)
                if update == e - 1:
                    self.avgs.append({n: t.clone() for n, t in self.ema.items()})

    def state_dict(self):
        return dict(opt=self.opt.state_dict(), ema=self.ema, ema_tag=self.ema_tag, avgs=self.avgs)

    def load_state_dict(self, state):
        self.opt.load_state_dict(state["opt"])
        self.ema, self.ema_tag, self.avgs = state["ema"], state["ema_tag"], state["avgs"]


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
