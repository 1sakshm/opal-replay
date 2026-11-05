"""A backend that replays programs instead of asking an agent: no agent, no cost.

At each decision OPAL asks it, it copies into the workspace the program that the plan says should be running by that
update, so OPAL's own swap machinery (trial steps, state carry-over via `previous`) runs exactly as with an agent.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from .records import RunRecord

_UPDATE = re.compile(r"Requested at update ([\d,]+)")
PROGRAM_FILES = ("updater.py", "analysis.py")


def replay_plan(record: RunRecord, mode: str, first_check: int) -> dict[int, Path]:
    """update -> program folder to install at the decision on that update.

    replay: every recorded switch, at its recorded update.
    final:  only the last version, at the first check (as early as OPAL allows a swap).
    """
    if mode == "replay":
        return {s.applied_at: record.version_dir(s.version) for s in record.switches}
    if mode == "final":
        return {first_check: record.version_dir(record.final_version)} if record.switches else {}
    raise ValueError(f"unknown mode {mode!r}")


def program_plan(program: Path, first_check: int) -> dict[int, Path]:
    """A single program folder (a recorded version or a hand-made variant), installed at the first check."""
    for name in PROGRAM_FILES:
        if not (program / name).exists():
            raise FileNotFoundError(f"{program} has no {name}")
    return {first_check: program}


class ReplayAgent:
    """Callable for opal.wrap(backend=...): fn(workspace, message, system) -> rationale."""

    def __init__(self, plan: dict[int, Path]):
        self.plan = dict(sorted(plan.items()))
        self.installed: Path | None = None  # program currently in the workspace (None: the starting one)
        self.log: list[tuple[int, str]] = []  # (update, program folder) actually installed

    def __call__(self, workspace: Path, message: str, system: str | None) -> str:
        m = _UPDATE.search(message)
        if not m:
            raise RuntimeError(f"replay: cannot read the update from the decision message: {message[:200]!r}")
        update = int(m.group(1).replace(",", ""))
        due = [p for u, p in self.plan.items() if u <= update]
        want = due[-1] if due else self.installed
        if want == self.installed:
            return f"replay: keep {self.installed} at update {update}"
        for name in PROGRAM_FILES:
            shutil.copyfile(want / name, Path(workspace) / name)
        self.installed = want
        self.log.append((update, str(want)))
        return f"replay: installed {want} at update {update}"
