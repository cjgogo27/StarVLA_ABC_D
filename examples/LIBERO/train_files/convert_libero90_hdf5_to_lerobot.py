"""Convert raw LIBERO-90 HDF5 demos to the LeRobot v2.1 layout used by starVLA.

Input layout:
    <raw-root>/
        *_demo.hdf5
        meta/modality.json

Each HDF5 file is one LIBERO task and contains ``data/demo_*`` trajectories.
This script writes one LeRobot dataset directory containing all task files and
all demos as individual episodes:

    <out-root>/<dataset-name>/
        meta/info.json
        meta/episodes.jsonl
        meta/tasks.jsonl
        meta/modality.json
        meta/embodiment.json
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/observation.images.image/episode_000000.mp4
        videos/chunk-000/observation.images.wrist_image/episode_000000.mp4

The produced modality names match ``examples/LIBERO/train_files/data_registry``:
    video.primary_image -> observation.images.image
    video.wrist_image   -> observation.images.wrist_image
    state.{x,y,z,roll,pitch,yaw,pad,gripper} -> observation.state[0:8]
    action.{x,y,z,roll,pitch,yaw,gripper}    -> action[0:7]

Usage:
    python examples/LIBERO/train_files/convert_libero90_hdf5_to_lerobot.py \\
        --raw-root ../data/LIBERO-datasets/libero_90 \\
        --out-root ../data/LIBERO-datasets/lerobot \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
VIDEO_KEYS = {
    "image": "agentview_rgb",
    "wrist_image": "eye_in_hand_rgb",
}
DEFAULT_DATASET_NAME = "libero_90_no_noops_1.0.0_lerobot"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _task_name_from_file(path: Path) -> str:
    return re.sub(r"_demo$", "", path.stem)


def _read_task_instruction(h5_data_group: h5py.Group, fallback: str) -> str:
    raw = _safe_decode(h5_data_group.attrs.get("problem_info", ""))
    if raw:
        try:
            info = json.loads(raw)
            instruction = info.get("language_instruction")
            if instruction:
                return str(instruction)
        except json.JSONDecodeError:
            pass
    return fallback.replace("_", " ")


def _build_state(demo: h5py.Group) -> np.ndarray:
    """Build starVLA LIBERO state: ee_pos(3), ee_ori(3), pad(1), gripper(1)."""
    obs = demo["obs"]
    ee_pos = np.asarray(obs["ee_pos"], dtype=np.float32)
    ee_ori = np.asarray(obs["ee_ori"], dtype=np.float32)
    gripper_states = np.asarray(obs["gripper_states"], dtype=np.float32)
    gripper = gripper_states[:, :1]
    pad = np.zeros_like(gripper, dtype=np.float32)
    return np.concatenate([ee_pos, ee_ori, pad, gripper], axis=1).astype(np.float32)


def _build_features(fps: int, height: int, width: int) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for key in VIDEO_KEYS:
        original_key = f"observation.images.{key}"
        features[original_key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    features["observation.state"] = {
        "dtype": "float32",
        "shape": [len(STATE_KEYS)],
        "names": ["state"],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [len(ACTION_KEYS)],
        "names": ["actions"],
    }
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
    return features


def _build_modality_json() -> dict[str, Any]:
    return {
        "state": {
            key: {
                "original_key": "observation.state",
                "start": i,
                "end": i + 1,
                "dtype": "float32",
            }
            for i, key in enumerate(STATE_KEYS)
        },
        "action": {
            key: {
                "original_key": "action",
                "start": i,
                "end": i + 1,
                "dtype": "float32",
                "absolute": False,
            }
            for i, key in enumerate(ACTION_KEYS)
        },
        "video": {
            "primary_image": {"original_key": "observation.images.image"},
            "wrist_image": {"original_key": "observation.images.wrist_image"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def _write_parquet(
    path: Path,
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    episode_index: int,
    task_index: int,
    global_offset: int,
) -> int:
    n = int(action.shape[0])
    table = pa.table(
        {
            "observation.state": pa.array(
                [row.tolist() for row in state],
                type=pa.list_(pa.float32(), len(STATE_KEYS)),
            ),
            "action": pa.array(
                [row.tolist() for row in action],
                type=pa.list_(pa.float32(), len(ACTION_KEYS)),
            ),
            "timestamp": pa.array(timestamp.astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
            "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64), type=pa.int64()),
            "index": pa.array(np.arange(global_offset, global_offset + n, dtype=np.int64), type=pa.int64()),
            "task_index": pa.array(np.full(n, task_index, dtype=np.int64), type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return n


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB video frames shaped [T,H,W,3], got {frames.shape}")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame in frames:
            rgb = np.asarray(frame, dtype=np.uint8)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _episode_sort_key(name: str) -> int:
    match = re.search(r"demo_(\d+)$", name)
    return int(match.group(1)) if match else 10**9


def convert_libero90(
    raw_root: Path,
    out_root: Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    fps: int = 20,
    overwrite: bool = False,
    limit_files: int | None = None,
    limit_episodes_per_file: int | None = None,
) -> Path:
    raw_root = raw_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw LIBERO-90 root not found: {raw_root}")

    hdf5_files = sorted(raw_root.glob("*.hdf5"))
    if limit_files is not None:
        hdf5_files = hdf5_files[:limit_files]
    if not hdf5_files:
        raise RuntimeError(f"No .hdf5 files found under {raw_root}")

    dataset_dir = out_root / dataset_name
    if dataset_dir.exists():
        if overwrite:
            print(f"[convert] removing existing output: {dataset_dir}")
            shutil.rmtree(dataset_dir)
        else:
            raise FileExistsError(f"Output already exists: {dataset_dir} (pass --overwrite)")

    (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "videos").mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    episodes: list[str] = []
    total_frames = 0
    total_videos = 0
    episode_index = 0
    first_height: int | None = None
    first_width: int | None = None

    for task_index, hdf5_path in enumerate(hdf5_files):
        task_name = _task_name_from_file(hdf5_path)
        with h5py.File(hdf5_path, "r") as f:
            data = f["data"]
            instruction = _read_task_instruction(data, task_name)
            tasks.append({"task_index": task_index, "task": instruction})

            demo_names = sorted(
                [name for name in data.keys() if name.startswith("demo_")],
                key=_episode_sort_key,
            )
            if limit_episodes_per_file is not None:
                demo_names = demo_names[:limit_episodes_per_file]

            for demo_name in demo_names:
                demo = data[demo_name]
                action = np.asarray(demo["actions"], dtype=np.float32)
                state = _build_state(demo)
                n = min(action.shape[0], state.shape[0])
                if n <= 0:
                    print(f"[convert][skip] empty {hdf5_path.name}:{demo_name}")
                    continue
                action = action[:n]
                state = state[:n]
                timestamp = np.arange(n, dtype=np.float32) / float(fps)

                agent_frames = np.asarray(demo["obs"][VIDEO_KEYS["image"]][:n], dtype=np.uint8)
                wrist_frames = np.asarray(demo["obs"][VIDEO_KEYS["wrist_image"]][:n], dtype=np.uint8)
                if first_height is None or first_width is None:
                    first_height, first_width = int(agent_frames.shape[1]), int(agent_frames.shape[2])

                chunk_index = episode_index // 1000
                parquet_path = dataset_dir / f"data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
                _write_parquet(
                    parquet_path,
                    state=state,
                    action=action,
                    timestamp=timestamp,
                    episode_index=episode_index,
                    task_index=task_index,
                    global_offset=total_frames,
                )

                for video_key, frames in {
                    "observation.images.image": agent_frames,
                    "observation.images.wrist_image": wrist_frames,
                }.items():
                    video_path = (
                        dataset_dir
                        / f"videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4"
                    )
                    _write_video(video_path, frames, fps=fps)
                    total_videos += 1

                episodes.append(
                    json.dumps(
                        {
                            "episode_index": episode_index,
                            "tasks": [instruction],
                            "length": n,
                        }
                    )
                )
                total_frames += n
                episode_index += 1

        print(
            f"[convert] {task_index + 1:03d}/{len(hdf5_files):03d} {hdf5_path.name}: "
            f"episodes={episode_index}, frames={total_frames}"
        )

    if first_height is None or first_width is None:
        raise RuntimeError("No valid episodes were converted.")

    info = {
        "codebase_version": "v2.1",
        "robot_type": "Panda",
        "total_episodes": episode_index,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": total_videos,
        "total_chunks": (episode_index + 999) // 1000,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{episode_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _build_features(fps=fps, height=first_height, width=first_width),
    }

    (dataset_dir / "meta/info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")
    (dataset_dir / "meta/tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks),
        encoding="utf-8",
    )
    (dataset_dir / "meta/episodes.jsonl").write_text("\n".join(episodes) + "\n", encoding="utf-8")
    (dataset_dir / "meta/modality.json").write_text(
        json.dumps(_build_modality_json(), indent=4),
        encoding="utf-8",
    )
    (dataset_dir / "meta/embodiment.json").write_text(
        json.dumps(
            {
                "robot_name": "Panda",
                "robot_type": "Panda",
                "record_frequency": fps,
                "body_controller_frequency": fps,
                "hand_controller_frequency": fps,
                "embodiment_tag": "franka",
            },
            indent=4,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    print(f"[convert] done: {dataset_dir}")
    print(f"[convert] tasks={len(tasks)}, episodes={episode_index}, frames={total_frames}, videos={total_videos}")
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/LIBERO-datasets/libero_90"))
    parser.add_argument("--out-root", type=Path, default=Path("data/LIBERO-datasets/lerobot"))
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None, help="Debug: convert only the first N HDF5 files.")
    parser.add_argument(
        "--limit-episodes-per-file",
        type=int,
        default=None,
        help="Debug: convert only the first N demos from each HDF5 file.",
    )
    args = parser.parse_args()

    convert_libero90(
        raw_root=args.raw_root,
        out_root=args.out_root,
        dataset_name=args.dataset_name,
        fps=args.fps,
        overwrite=args.overwrite,
        limit_files=args.limit_files,
        limit_episodes_per_file=args.limit_episodes_per_file,
    )


if __name__ == "__main__":
    main()
