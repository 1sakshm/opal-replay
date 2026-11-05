"""Build ablation programs for the MNIST diffusion task (record op-diff-499-b) into variants/diff/.

Each variant is a folder with updater.py + analysis.py, run with
    python -m opal_replay.run --mode program --program variants/diff/<name> --out runs/diff/<name>-s0 \
        third_party/opal/examples/diffusion_mnist.py --data_dir data/datasets --seed 0

Variants of the agent's final program (v014: Muon + AdamW, warmup-hold-decay schedule, per-layer LR multipliers):
    final-no-layer-mult    v014 with every per-layer multiplier at 1
    final-planned-decay    v014 with the decay it had planned before re-timing it at update 2000 (0.5 -> 1.0, to 0.02)
Human-obvious baselines (no Muon; the script's own AdamW, the agent's schedule shape):
    adamw-wsd-lr2e-4       the script's AdamW at its own lr 2e-4, v014's schedule
    adamw-wsd-lr1e-3       the same at lr 1e-3
The agent's first program (textbook Muon + schedule, written at update 50) needs no variant: use its recorded folder.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal_replay import records  # noqa: E402

OUT = ROOT / "variants" / "diff"

ADAMW_WSD = '''"""Updater (variant adamw-wsd): the training script's own AdamW, with the warmup-hold-decay schedule of the agent's
final program (v014) in place of the script's warmup-then-constant one. No Muon, no per-layer multipliers."""

import torch

BASE_LR = {lr}
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
'''


def sub(text, pattern, repl):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"pattern not found: {pattern}")
    return new


def write(name, updater, analysis_src):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "updater.py").write_text(updater, encoding="utf-8")
    shutil.copyfile(analysis_src, d / "analysis.py")
    print("wrote", d.relative_to(ROOT))


def main():
    rec = records.load(ROOT / "data/opal-results/small_scripts/op-diff-499-b")
    final = rec.version_dir(rec.final_version)
    v014 = (final / "updater.py").read_text(encoding="utf-8")

    write("final-no-layer-mult", sub(v014, r"^LR_MULT = .*$", "LR_MULT = ()"), final / "analysis.py")

    planned = sub(v014, r"^DECAY_START_FRAC = .*$", "DECAY_START_FRAC = 0.5")
    planned = sub(planned, r"^DECAY_END_FRAC = .*$", "DECAY_END_FRAC = 1.0")
    planned = sub(planned, r"^FINAL_FACTOR = .*$", "FINAL_FACTOR = 0.02")
    write("final-planned-decay", planned, final / "analysis.py")

    start_analysis = rec.version_dir(0) / "analysis.py"  # the generic starting analysis, no agent edits
    for lr in ("2e-4", "1e-3"):
        write(f"adamw-wsd-lr{lr}", ADAMW_WSD.format(lr=lr), start_analysis)

    print(f"agent's first program: {rec.version_dir(1).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
