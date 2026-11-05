"""Read an OPAL run record: the program versions the agent wrote and the update at which each took over."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Switch:
    version: int  # programs/v{version:03d}
    applied_at: int  # first update the program ran on
    requested_at: int
    decision: int  # k, the decision that produced it


@dataclass
class RunRecord:
    root: Path  # the folder holding result.json
    run_dir: Path  # the OPAL run folder holding events.jsonl and programs/
    total_updates: int
    blocking: bool
    schedule: list[int]
    switches: list[Switch]
    result: dict = field(default_factory=dict)

    @property
    def programs(self) -> Path:
        return self.run_dir / "programs"

    def version_dir(self, version: int) -> Path:
        return self.programs / f"v{version:03d}"

    @property
    def final_version(self) -> int:
        return self.switches[-1].version if self.switches else 0

    def version_at(self, update: int) -> int:
        """The program version that ran at this update in the original run."""
        v = 0
        for s in self.switches:
            if s.applied_at <= update:
                v = s.version
        return v


def _find_run_dir(root: Path) -> Path:
    hits = sorted(p.parent for p in root.rglob("events.jsonl") if (p.parent / "programs").is_dir())
    if not hits:
        raise FileNotFoundError(f"no OPAL run (events.jsonl next to programs/) under {root}")
    if len(hits) > 1:
        raise ValueError(f"several OPAL runs under {root}; pass one of: {[str(h) for h in hits]}")
    return hits[0]


def load(path: str | Path) -> RunRecord:
    """`path` is a run-record folder (e.g. data/opal-results/small_scripts/op-gpt-499-b) or an OPAL run folder."""
    root = Path(path)
    run_dir = _find_run_dir(root)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    start = next(e for e in events if e["kind"] == "start")
    switches = [
        Switch(e["version"], e["applied_at"], e.get("requested_at", e["applied_at"]), e.get("k", -1))
        for e in events
        if e["kind"] == "applied"
    ]
    for s in switches:
        if not (run_dir / "programs" / f"v{s.version:03d}" / "updater.py").exists():
            raise FileNotFoundError(f"{root}: version {s.version} was applied but its program is not in the record")
    result_path = next((p for p in [root / "result.json", *root.rglob("result.json")] if p.exists()), None)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path else {}
    return RunRecord(
        root=root,
        run_dir=run_dir,
        total_updates=start["total_updates"],
        blocking=bool(start.get("blocking", True)),
        schedule=list(start.get("schedule") or []),
        switches=switches,
        result=result,
    )
