#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def get_chain_sr(summary: dict, index: int) -> float:
    chain_sr = summary["chain_sr"]
    return float(chain_sr.get(str(index), chain_sr.get(index, 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    args = parser.parse_args()

    result_files = sorted(args.eval_root.glob("worker_*/results.json"))
    if not result_files:
        raise FileNotFoundError(f"No worker results found under {args.eval_root}/worker_*/results.json")

    results = []
    task_success = Counter()
    task_total = Counter()
    for path in result_files:
        data = json.loads(path.read_text())
        run = data.get("0") or data.get(0)
        if run is None:
            raise KeyError(f"Missing epoch 0 in {path}")
        for task, info in run.get("task_info", {}).items():
            task_success[task] += int(info["success"])
            task_total[task] += int(info["total"])
        seq_file = path.parent / "sequence_results.json"
        if seq_file.exists():
            results.extend(json.loads(seq_file.read_text())["results"])

    if results:
        count = Counter(results)
        chain_sr = {}
        for i in range(1, 6):
            chain_sr[str(i)] = sum(count[j] for j in reversed(range(i, 6))) / len(results)
        avg_seq_len = float(np.mean(results))
        num_sequences = len(results)
    else:
        # Fallback: average worker summaries weighted equally if sequence files are absent.
        summaries = [(json.loads(path.read_text()).get("0") or json.loads(path.read_text()).get(0)) for path in result_files]
        avg_seq_len = float(np.mean([s["avg_seq_len"] for s in summaries]))
        chain_sr = {str(i): float(np.mean([get_chain_sr(s, i) for s in summaries])) for i in range(1, 6)}
        num_sequences = None

    task_info = {
        task: {
            "success": task_success[task],
            "total": task_total[task],
            "sr": task_success[task] / task_total[task] if task_total[task] else 0.0,
        }
        for task in sorted(task_total)
    }
    merged = {
        "num_workers": len(result_files),
        "num_sequences": num_sequences,
        "avg_seq_len": avg_seq_len,
        "chain_sr": chain_sr,
        "task_info": task_info,
        "worker_result_files": [str(path) for path in result_files],
    }
    out = args.eval_root / "merged_results.json"
    out.write_text(json.dumps(merged, indent=2))
    print(json.dumps(merged, indent=2))
    print(f"merged results saved to {out}")


if __name__ == "__main__":
    main()
