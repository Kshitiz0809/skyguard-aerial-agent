"""Download a handful of real VisDrone2019-DET frames from the Hugging Face Hub.

    python scripts/download_visdrone.py --n 6

Images land in data/samples/visdrone/. They have no ground-truth sidecar, so use
a real VLM backend (groq / qwen_local / hf_api) to analyse them.
Dataset: https://huggingface.co/datasets/Voxel51/VisDrone2019-DET (see its licence).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
REPO = "Voxel51/VisDrone2019-DET"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "samples" / "visdrone")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = []
    for item in api.list_repo_tree(REPO, path_in_repo="data", repo_type="dataset"):
        if item.path.lower().endswith(".jpg"):
            files.append(item.path)
        if len(files) >= args.n * 40:
            break
    picks = files[:: max(1, len(files) // args.n)][: args.n]  # spread across sequences
    for path in picks:
        local = hf_hub_download(REPO, path, repo_type="dataset")
        dest = args.out / Path(path).name
        shutil.copyfile(local, dest)
        print(f"saved {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
