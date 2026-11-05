"""Updater: Muon for hidden conv / linear weight matrices, AdamW for everything else (biases, GroupNorm gains,
the input conv and the output conv). Own warmup-stable-decay schedule as a function of the absolute update index.

Muon: momentum (EMA form, nesterov), Newton-Schulz orthogonalisation in bfloat16 on the weight reshaped to
(out, in*kh*kw), scaled by sqrt(max(1, out/in)) as in the original Muon. Same-shape matrices are stacked.
"""

import copy

import torch

# ----- settings -----------------------------------------------------------------------------------------------------
MUON_LR = 0.02  # peak
MUON_MOMENTUM = 0.95
MUON_WD = 0.0
ADAM_LR = 1e-3  # peak
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WD = 0.0

WARMUP_END = 200  # lr factor = min(1, update / WARMUP_END) during warmup
DECAY_START_FRAC = 0.5  # constant until here, then linear down to FINAL_FACTOR at the last update
FINAL_FACTOR = 0.02

ADAM_ONLY = ("inp.weight", "out.2.weight")  # input / output layers stay on AdamW


def lr_factor(update, total):
    w = min(1.0, update / WARMUP_END)
    s = DECAY_START_FRAC * total
    if update <= s:
        return w
    x = min(1.0, (update - s) / max(1.0, total - s))
    return w * (1.0 - x * (1.0 - FINAL_FACTOR))


def is_muon(name, p):
    return p.ndim >= 2 and name not in ADAM_ONLY


@torch.no_grad()
def newton_schulz(G, steps=5, eps=1e-7):
    """Batched quintic Newton-Schulz (Keller Jordan's coefficients), in bfloat16. G: (B, m, n)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X


class Updater:
    follow_user_scheduler = False
    can_skip = False

    def __init__(self, named_params, previous=None, spec=None):
        spec = spec or {}
        self.total = spec.get("total_updates", 6000)
        self.named = named_params
        self.muon_named = [(n, p) for n, p in named_params if is_muon(n, p)]
        self.adam_named = [(n, p) for n, p in named_params if not is_muon(n, p)]

        self.adam = torch.optim.AdamW(
            [p for _, p in self.adam_named], lr=ADAM_LR, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=ADAM_WD,
            foreach=True,
        )
        self.muon_group = dict(params=[p for _, p in self.muon_named], lr=MUON_LR, name="muon")

        # shape buckets for batched Newton-Schulz
        self.buckets = {}
        for n, p in self.muon_named:
            shape = (p.shape[0], p[0].numel())
            self.buckets.setdefault(shape, []).append((n, p))

        self.muon_buf = {n: torch.zeros_like(p) for n, p in self.muon_named}
        # device-side statistics for analysis
        self.stats = {}
        self._carry(previous, spec)

    # ----- carry state ------------------------------------------------------------------------------------------
    def _carry(self, previous, spec):
        if previous is None:
            return
        by_param = {}  # id(param) -> AdamW state of the previous program
        prev_bufs = getattr(previous, "muon_buf", {})
        if hasattr(previous, "adam"):
            by_param.update({id(p): st for p, st in previous.adam.state.items()})
        else:  # starting program: the training script's optimizers
            for o in getattr(previous, "opts", None) or spec.get("user_optimizers", []):
                by_param.update({id(p): st for p, st in o.state.items()})
        for n, p in self.muon_named:  # muon momentum: previous muon buffer, else AdamW's exp_avg
            st = by_param.get(id(p))
            if n in prev_bufs:
                self.muon_buf[n] = prev_bufs[n].detach().clone()
            elif st is not None and "exp_avg" in st:
                self.muon_buf[n] = st["exp_avg"].detach().clone()
        for n, p in self.adam_named:  # AdamW state carried; a tensor arriving from muon starts fresh
            st = by_param.get(id(p))
            if st is not None and "exp_avg" in st:
                self.adam.state[p] = copy.deepcopy(st)

    @property
    def param_groups(self):
        return list(self.adam.param_groups) + [self.muon_group]

    # ----- step -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, grads, update, ctx=None):
        f = lr_factor(update, self.total)
        for g in self.adam.param_groups:
            g["lr"] = ADAM_LR * f
        self.muon_group["lr"] = MUON_LR * f
        lr = MUON_LR * f

        # Muon
        for (m, k), items in self.buckets.items():
            gs = [grads[n] if grads[n] is not None else torch.zeros_like(p) for n, p in items]
            bufs = [self.muon_buf[n] for n, _ in items]
            torch._foreach_lerp_(bufs, gs, 1 - MUON_MOMENTUM)
            nest = torch._foreach_lerp(gs, bufs, MUON_MOMENTUM)
            G = torch.stack([x.reshape(m, k) for x in nest])
            O = newton_schulz(G).to(G.dtype)
            scale = max(1.0, m / k) ** 0.5
            ps = [p for _, p in items]
            if MUON_WD:
                torch._foreach_mul_(ps, 1 - lr * MUON_WD)
            upd = [O[i].reshape(p.shape) for i, p in enumerate(ps)]
            torch._foreach_add_(ps, upd, alpha=-lr * scale)

        # AdamW
        self.adam.step()

    # ----- state ------------------------------------------------------------------------------------------------
    def state_dict(self):
        return dict(adam=self.adam.state_dict(), muon_buf={n: b for n, b in self.muon_buf.items()})

    def load_state_dict(self, state):
        self.adam.load_state_dict(state["adam"])
        for g in self.adam.param_groups:
            g["betas"], g["eps"], g["weight_decay"] = ADAM_BETAS, ADAM_EPS, ADAM_WD
        for n, b in state["muon_buf"].items():
            if n in self.muon_buf:
                self.muon_buf[n].copy_(b)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
