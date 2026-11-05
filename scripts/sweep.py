"""Run plain / replay / final over several seeds for one recorded run, then print a summary table.

    python scripts/sweep.py --task diff --seeds 0 1 2
    python scripts/sweep.py --task mnist --seeds 0 1 2 3 4 --modes plain replay
    python scripts/sweep.py --task diff --summary-only
    python scripts/sweep.py --task diff --modes program:variants/diff/adamw-wsd-lr1e-3 program:<recorded v001 folder>

A mode `program:<folder>` runs that one program (see scripts/make_variants.py); its runs are named after the folder.

Finished runs (with a result.json) are skipped, so an interrupted sweep can be restarted.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EX = "third_party/opal/examples"
REC = "data/opal-results"

# task -> (record, script, script args, metric key in result.json, higher_is_better, script --out kind)
TASKS = {
    "mnist": (f"{REC}/small_scripts/mnist-opal", f"{EX}/mnist_mlp.py", ["--data_dir", "data/datasets"], "test_accuracy", True, "file"),
    "diff": (f"{REC}/small_scripts/op-diff-499-b", f"{EX}/diffusion_mnist.py", ["--data_dir", "data/datasets"], "validation_loss_ema", False, "file"),
    "cifar": (f"{REC}/small_scripts/op-cifar-499-b", f"{EX}/cifar10_resnet.py", ["--data_dir", "data/datasets/cifar10"], "test_accuracy", True, "file"),
    "gpt": (f"{REC}/small_scripts/op-gpt-499-b", f"{EX}/gpt_small.py", ["--data_dir", "data/datasets/fineweb10B-gpt2"], "validation_loss", False, "file"),
    "overfit": (f"{REC}/overfit/op-ofit-499-b", f"{EX}/overfit/train.py", ["--setting", "label10", "--lr", "3e-3", "--epochs", "120", "--data-dir", "data/datasets/cifar10"], "test_accuracy", True, "dir"),
}
# script arguments only OPAL-wrapped runs may take
OPAL_ARGS = {"overfit": ["--validation", "--decisions", "30", "--eval-seed", "0"]}
# script arguments that set the seed
SEED_ARGS = {"overfit": ["--seed", "{s}", "--data-seed", "{s}"]}


def metric(path, key):
    d = json.loads(path.read_text(encoding="utf-8"))
    if key in d:
        return d[key]
    m = d.get("metric")
    if isinstance(m, dict) and "value" in m:  # opal.results.summary(): {"name", "value", "direction"}
        return m["value"]
    return None


def label(mode):
    return Path(mode.split(":", 1)[1]).name if mode.startswith("program:") else mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+", default=["plain", "replay", "final"])
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    record, script, extra, key, higher, out_kind = TASKS[a.task]
    runs = ROOT / "runs" / a.task
    if not a.summary_only:
        for seed in a.seeds:
            for mode in a.modes:
                out = runs / f"{label(mode)}-s{seed}"
                if (out / "result.json").exists():
                    print(f"skip {out} (done)")
                    continue
                cmd = [sys.executable, "-m", "opal_replay.run", "--out", str(out), "--script-out", out_kind]
                if mode.startswith("program:"):
                    cmd += ["--mode", "program", "--program", mode.split(":", 1)[1]]
                else:
                    cmd += ["--mode", mode]
                if mode in ("replay", "final"):
                    cmd += ["--record", record]
                cmd += [script, *extra, *[x.format(s=seed) for x in SEED_ARGS.get(a.task, ["--seed", "{s}"])]]
                if mode != "plain":
                    cmd += OPAL_ARGS.get(a.task, [])
                print(">>", " ".join(cmd), flush=True)
                subprocess.run(cmd, cwd=ROOT, check=True)

    print(f"\n{a.task}: {key} ({'higher' if higher else 'lower'} is better)")
    for mode in a.modes:
        vals = []
        for p in sorted(runs.glob(f"{label(mode)}-s*/result.json")):
            v = metric(p, key)
            if v is not None:
                vals.append(v)
        if vals:
            sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            print(f"  {label(mode):22} n={len(vals)}  mean {statistics.mean(vals):.4f}  sd {sd:.4f}  {[round(v, 4) for v in vals]}")
        else:
            print(f"  {label(mode):22} no results")


if __name__ == "__main__":
    main()
