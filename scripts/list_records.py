"""Print what each downloaded OPAL run record holds: metric, OPAL vs static, and the program switches.

    python scripts/list_records.py [data/opal-results]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opal_replay import records  # noqa: E402


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "data/opal-results")
    for res in sorted(base.rglob("result.json")):
        root = res.parent
        try:
            rec = records.load(root)
        except (FileNotFoundError, ValueError):
            continue  # static runs, nested result.json files
        r = rec.result
        if r.get("role") != "OPAL":
            continue
        sw = ", ".join(f"v{s.version}@{s.applied_at}" for s in rec.switches)
        print(f"{r.get('run')}  [{r.get('example')}]  {r.get('metric')} ({r.get('direction')} is better)")
        print(f"   OPAL {r.get('value')}   static {r.get('baseline_value')} ({r.get('baseline_run')})")
        print(f"   {rec.total_updates} updates, blocking={rec.blocking}, gpu={r.get('hardware', {}).get('gpu')}")
        print(f"   switches: {sw or 'none'}")
        print(f"   command: {r.get('exact_command') or r.get('entry_command')}")
        print()


if __name__ == "__main__":
    main()
