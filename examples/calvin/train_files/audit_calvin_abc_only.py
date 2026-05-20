#!/usr/bin/env python
"""Audit the local CALVIN ABC->D training data before launching StarVLA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", required=True, type=Path)
    parser.add_argument("--lerobot-name", default="calvin_task_ABC_D")
    parser.add_argument("--original-name", default="task_ABC_D")
    args = parser.parse_args()

    shared_root = args.shared_root.resolve()
    lerobot_dir = shared_root / args.lerobot_name
    original_dir = shared_root / args.original_name
    original_training_dir = original_dir / "training"
    original_validation_dir = original_dir / "validation"

    info_path = lerobot_dir / "meta" / "info.json"
    episodes_path = lerobot_dir / "meta" / "episodes.jsonl"
    tasks_path = lerobot_dir / "meta" / "tasks.jsonl"

    required_paths = [
        shared_root,
        lerobot_dir,
        original_dir,
        original_training_dir,
        info_path,
        episodes_path,
        tasks_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required CALVIN ABC-only paths: " + ", ".join(missing))

    if original_validation_dir.exists():
        raise RuntimeError(
            f"Refusing to launch: found {original_validation_dir}. "
            "Calvin D validation/test data must not be present in the training source."
        )

    info = read_json(info_path)
    episode_lines = count_files(lerobot_dir / "data", "chunk-*/*.parquet")
    video_files = count_files(lerobot_dir / "videos", "chunk-*/*/*.mp4")
    original_npz_files = count_files(original_training_dir, "episode_*.npz")

    audit = {
        "compliance": "calvin_abc_only_no_d_training_data",
        "shared_root": str(shared_root),
        "lerobot_training_dataset": str(lerobot_dir),
        "original_training_source": str(original_training_dir),
        "original_validation_dir_present": original_validation_dir.exists(),
        "lerobot_info_total_episodes": info.get("total_episodes"),
        "lerobot_info_total_frames": info.get("total_frames"),
        "lerobot_info_splits": info.get("splits"),
        "lerobot_parquet_episodes": episode_lines,
        "lerobot_video_files": video_files,
        "original_training_npz_files": original_npz_files,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
