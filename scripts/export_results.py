"""Write every finished run's numbers to results/runs.csv, so the tables in the README can be checked without the runs/ folder.

    python scripts/export_results.py
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["task", "program", "seed", "metric", "value", "test_accuracy", "test_loss", "dev_accuracy", "dev_loss", "train_seconds"]


def row(task, path):
    program, seed = re.fullmatch(r"(.+)-s(\d+)", path.parent.name).groups()
    d = json.loads(path.read_text(encoding="utf-8"))
    final = d.get("final", {})
    return dict(
        task=task,
        program=program,
        seed=int(seed),
        metric=d["metric"]["name"],
        value=d["metric"]["value"],
        test_accuracy=final.get("test_acc", d.get("test_accuracy")),
        test_loss=final.get("test_loss", d.get("test_loss")),
        dev_accuracy=final.get("dev_acc"),
        dev_loss=final.get("dev_loss"),
        train_seconds=d.get("train_seconds"),
    )


def main():
    rows = [row(t, p) for t in ("mnist", "diff", "diff-floor", "overfit") for p in sorted((ROOT / "runs" / t).glob("*/result.json"))]
    rows.sort(key=lambda r: (r["task"], r["program"], r["seed"]))
    out = ROOT / "results" / "runs.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
