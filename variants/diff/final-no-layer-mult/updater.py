"""Updater: Muon for hidden conv / linear weight matrices, AdamW for everything else (biases, GroupNorm gains,
the input conv and the output conv). Own warmup-stable-decay schedule as a function of the absolute update index.

Muon: momentum (EMA form, nesterov), Newton-Schulz orthogonalisation in bfloat16 on the weight reshaped to
(out, in*kh*kw), scaled by sqrt(max(1, out/in)) as in the original Muon. Same-shape matrices are stacked.
"""

import copy

import torch

# ----- settings -----------------------------------------------------------------------------------------------------
MUON_LR = 0.02  # peak (0.02 to u200, 0.01 u200-300 on a misread cos signal, back to 0.02 at u300 as a test)
MUON_MOMENTUM = 0.95
MUON_WD = 0.0
ADAM_LR = 5e-4  # peak (was 1e-3 until u150: persistent cos(g,m) < -0.5 on biases/norms/io)
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WD = 0.0

WARMUP_END = 200  # lr factor = min(1, update / WARMUP_END) during warmup
# Decay: constant until DECAY_START_FRAC, linear down to FINAL_FACTOR at DECAY_END_FRAC, held there to the end.
# (Until u2000 the plan was start 0.5 / end 1.0 / 0.02; moved at u2000 because the constant-lr loss had been flat at
# ~0.175 for 600 updates. The final hold gives the EMA(0.999) ~300 low-lr updates to catch up.)
DECAY_START_FRAC = 2000 / 6000
DECAY_END_FRAC = 0.95
FINAL_FACTOR = 0.03

ADAM_ONLY = ("inp.weight", "out.2.weight")  # input / output layers stay on AdamW

# per-tensor lr multipliers by name prefix (first match). From u1000: the 28x28 layers (inp, d1, u1, out) carry the
# strongest step-to-step oscillation (cos(g,m) -0.6..-0.95 vs ~0 at 7x7); test whether halving only them lowers the
# loss floor.
# u1000-1150 at 0.5: 50-means 0.184 -> 0.1818, 0.1804, 0.1781, grad norm 0.30 -> 0.17 (floor lives there).
# From u1150: 0.25 -> 50-means 0.1797, 0.1776, 0.1781 (no further gain, no harm; kept).
# From u1300: the 14x14 ring (d2., u2.) at 0.5 -> 50-means 0.1804, 0.1768, 0.1748 (gain, kept).
# u1450-1600: 7x7 ring (d3., mid., u3.) at 0.5 -> 50-means 0.1756, 0.1758, 0.1748: no gain; reverted to 1 at u1600.
LR_MULT = ()


def lr_mult(name):
    for pre, m in LR_MULT:
        if name.startswith(pre):
            return m
    return 1.0


def lr_factor(update, total):
    w = min(1.0, update / WARMUP_END)
    s, e = DECAY_START_FRAC * total, DECAY_END_FRAC * total
    if update <= s:
        return w
    x = min(1.0, (update - s) / max(1.0, e - s))
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

        groups = {}
        for n, p in self.adam_named:
            groups.setdefault(lr_mult(n), []).append(p)
        self.adam = torch.optim.AdamW(
            [dict(params=ps, mult=m) for m, ps in sorted(groups.items())], lr=ADAM_LR, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=ADAM_WD,
            foreach=True,
        )
        self.muon_group = dict(params=[p for _, p in self.muon_named], lr=MUON_LR, name="muon")

        # one zero-padded batch for Newton-Schulz: every matrix as (rows <= cols), padded to (R, C); zero padding
        # leaves the result on each matrix's own block unchanged
        self.mu_meta = []  # (rows, cols, transposed, scale)
        for n, p in self.muon_named:
            m, k = p.shape[0], p[0].numel()
            t = m > k
            self.mu_meta.append((min(m, k), max(m, k), t, max(1.0, m / k) ** 0.5 * lr_mult(n)))
        R = max(r for r, _, _, _ in self.mu_meta)
        C = max(c for _, c, _, _ in self.mu_meta)
        dev = self.muon_named[0][1].device
        self.mu_pad = torch.zeros(len(self.muon_named), R, C, dtype=torch.bfloat16, device=dev)

        self.muon_buf = {n: torch.zeros_like(p) for n, p in self.muon_named}
        # device-side statistics for analysis
        self.stats = {}  # class -> 0-d device tensor: EMA over updates of cos(g_t, m_{t-1})
        self.stat_classes = {}
        for n, p in self.adam_named:
            c = "norm" if ("norm" in n or n.startswith("out.0")) else ("bias" if n.endswith(".bias") else "io")
            self.stat_classes.setdefault(c, []).append((n, p))
        self._carry(previous, spec)

    # ----- carry state ------------------------------------------------------------------------------------------
    def _carry(self, previous, spec):
        if previous is None:
            return
        self.stats = {c: t.detach().clone() for c, t in getattr(previous, "stats", {}).items()}
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
            g["lr"] = ADAM_LR * f * g["mult"]
        self.muon_group["lr"] = MUON_LR * f
        lr = MUON_LR * f

        # diagnostics: cosine of this gradient with the momentum before this update, per class (device only)
        mg = [grads[n] for n, _ in self.muon_named if grads[n] is not None]
        mm = [self.muon_buf[n] for n, _ in self.muon_named if grads[n] is not None]
        self._cos_stat("muon", mg, mm)
        for c, items in self.stat_classes.items():
            pairs = [(grads[n], self.adam.state[p]["exp_avg"]) for n, p in items
                     if grads[n] is not None and p in self.adam.state]
            if pairs:
                self._cos_stat(c, [a for a, _ in pairs], [b for _, b in pairs])

        # Muon
        ps = [p for _, p in self.muon_named]
        gs = [grads[n] if grads[n] is not None else torch.zeros_like(p) for n, p in self.muon_named]
        bufs = [self.muon_buf[n] for n, _ in self.muon_named]
        torch._foreach_lerp_(bufs, gs, 1 - MUON_MOMENTUM)
        nest = torch._foreach_lerp(gs, bufs, MUON_MOMENTUM)
        G = self.mu_pad
        for i, (x, (r, c, t, _)) in enumerate(zip(nest, self.mu_meta)):
            x2 = x.reshape(x.shape[0], -1)
            G[i, :r, :c].copy_(x2.mT if t else x2)
        O = newton_schulz(G).float()
        upd = []
        for i, (p, (r, c, t, _)) in enumerate(zip(ps, self.mu_meta)):
            o = O[i, :r, :c]
            upd.append((o.mT if t else o).reshape(p.shape))
        if MUON_WD:
            torch._foreach_mul_(ps, 1 - lr * MUON_WD)
        upd = torch._foreach_mul(upd, [s for _, _, _, s in self.mu_meta])
        torch._foreach_add_(ps, upd, alpha=-lr)

        # AdamW
        self.adam.step()

    def _cos_stat(self, c, gs, ms):
        if not gs:
            return
        g2 = torch.stack(torch._foreach_norm(gs)).square().sum()
        m2 = torch.stack(torch._foreach_norm(ms)).square().sum()
        s2 = torch.stack(torch._foreach_norm(torch._foreach_add(gs, ms))).square().sum()
        cos = (s2 - g2 - m2) / (2 * (g2 * m2).sqrt() + 1e-30)
        snr = m2 / (g2 + 1e-30)  # |m|^2/|g|^2: ~(1-b)/(1+b) if the gradients were pure noise, ~1 if pure signal
        for key, val in ((c, cos), (c + "_m2g2", snr)):
            if key not in self.stats:
                self.stats[key] = torch.zeros((), device=cos.device)
            self.stats[key].mul_(0.95).add_(val, alpha=0.05)

    # ----- state ------------------------------------------------------------------------------------------------
    def state_dict(self):
        return dict(adam=self.adam.state_dict(), muon_buf={n: b for n, b in self.muon_buf.items()})

    def load_state_dict(self, state):
        mults = [g["mult"] for g in self.adam.param_groups]
        self.adam.load_state_dict(state["adam"])
        for g, m in zip(self.adam.param_groups, mults):
            g["mult"] = m
            g["betas"], g["eps"], g["weight_decay"] = ADAM_BETAS, ADAM_EPS, ADAM_WD
        for n, b in state["muon_buf"].items():
            if n in self.muon_buf:
                self.muon_buf[n].copy_(b)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
