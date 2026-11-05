"""Run an OPAL example script with a recorded run's programs replayed instead of an agent. No API calls.

    python -m opal_replay.run --record data/opal-results/small_scripts/op-diff-499-b --mode replay \
        --out runs/diff/replay-s1 -- third_party/opal/examples/diffusion_mnist.py --seed 1

Modes:
    replay   install each recorded program version at the update where the original run applied it
    final    install only the last recorded version, at the first check
    program  install one program folder (--program: a recorded version or a variant), at the first check
    keep     OPAL wrapped, no changes ever (must match the plain run; a sanity check)
    plain    the script without OPAL (the static recipe)

The script's own arguments go after `--`. This module adds --opal/--backend itself; do not pass them.
Results: the script's --out JSON (if it has one) goes to <out>/result.json, OPAL's records to <out>/opal,
and <out>/replay.json says what was installed when.
"""
from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
import time
from pathlib import Path

from . import records
from .backend import ReplayAgent, program_plan, replay_plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", help="run-record folder (needed for replay/final)")
    ap.add_argument("--mode", choices=["replay", "final", "program", "keep", "plain"], required=True)
    ap.add_argument("--program", help="folder with updater.py and analysis.py (for --mode program)")
    ap.add_argument("--out", required=True, help="output folder for this run")
    ap.add_argument("--every", type=int, default=50, help="OPAL check interval; must match the script's --every")
    ap.add_argument(
        "--script-out",
        choices=["file", "dir", "none"],
        default="file",
        help="what the script's --out takes: a JSON file (examples/*.py), a folder (examples/overfit/train.py), or none",
    )
    ap.add_argument("script")
    ap.add_argument("script_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)

    script_args = [x for x in a.script_args if x != "--"]
    if any(x in ("--opal", "--backend") for x in script_args):
        ap.error("do not pass --opal/--backend to the script; --mode sets them")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(mode=a.mode, record=a.record, program=a.program, script=a.script, script_args=script_args, started=time.time())

    agent = None
    if a.mode in ("replay", "final"):
        if not a.record:
            ap.error(f"--mode {a.mode} needs --record")
        rec = records.load(a.record)
        plan = replay_plan(rec, a.mode, first_check=a.every)
        agent = ReplayAgent(plan)
        meta.update(plan={str(u): str(v) for u, v in plan.items()}, original_schedule=rec.schedule)
        _patch_wrap(agent, sorted(set(rec.schedule) | set(plan)), out / "opal")
    elif a.mode == "program":
        if not a.program:
            ap.error("--mode program needs --program")
        plan = program_plan(Path(a.program).resolve(), first_check=a.every)
        agent = ReplayAgent(plan)
        meta.update(plan={str(u): str(v) for u, v in plan.items()})
        _patch_wrap(agent, sorted(plan), out / "opal")
    elif a.mode == "keep":
        _patch_wrap(None, None, out / "opal")

    argv_script = [a.script, *script_args]
    if a.mode != "plain":
        argv_script += ["--opal", "--backend", "keep"]  # the backend is replaced inside wrap
    if "--out" not in script_args and a.script_out != "none":
        argv_script += ["--out", str(out / "result.json" if a.script_out == "file" else out / "script")]

    sys.argv = argv_script
    try:
        runpy.run_path(a.script, run_name="__main__")
        if a.script_out == "dir" and (out / "script" / "result.json").exists():
            shutil.copyfile(out / "script" / "result.json", out / "result.json")
    finally:
        meta.update(ended=time.time(), installed=agent.log if agent else [])
        (out / "replay.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def _patch_wrap(agent, schedule, run_dir):
    import opal

    original = opal.wrap

    def wrap(*args, **kw):
        kw["run_dir"] = str(run_dir)
        kw["summarize"] = False
        kw["blocking"] = True  # a replay answers instantly; this keeps applied_at == requested update
        if agent is not None:
            kw["backend"] = agent
            kw["schedule"] = [u for u in schedule if u <= kw["total_steps"]]  # a shortened run ends before some
            kw.pop("decisions", None)
        else:
            kw["backend"] = "keep"
        return original(*args, **kw)

    opal.wrap = wrap


if __name__ == "__main__":
    main()
