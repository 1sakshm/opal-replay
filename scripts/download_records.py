"""Download OPAL run records (logs + agent-written programs, no model weights).

Source: https://huggingface.co/datasets/snow-leopard/opal-results (MIT license).

Usage:
    python scripts/download_records.py                 # small_scripts + overfit
    python scripts/download_records.py --groups goldfish raev2_b
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "snow-leopard/opal-results"
DEFAULT_GROUPS = ["small_scripts", "overfit"]
WEIGHT_PATTERNS = ["*.pt", "*.pth", "*.safetensors", "*.bin", "*.ckpt", "*.npz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    ap.add_argument("--out", default="data/opal-results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=out,
        allow_patterns=[f"{g}/**" for g in args.groups] + ["README.md", "LICENSE"],
        ignore_patterns=WEIGHT_PATTERNS,
    )
    print(f"downloaded to {path}")


if __name__ == "__main__":
    main()
