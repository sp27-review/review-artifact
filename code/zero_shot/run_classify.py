#!/usr/bin/env python3
"""(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


LABEL_MAP = {
    "UNCLASSIFIED": "Unclassified",
    "CONFIDENTIAL": "Confidential",
    "SECRET": "Secret",
    "TOP SECRET": "Top Secret",
}


from corelib.paths import (
    EVIDENCE_RESULTS_DIR,
    DISC_EXPERIMENT_DIR,
    LLAMA_STACK_CONFIG as DEFAULT_LLAMA_STACK_CONFIG,
)
from corelib.workspace import resolve_workspace_path

DISC_RUNNER = DISC_EXPERIMENT_DIR / "scripts" / "disc_classify_experiment.py"


def default_results_dir(input_path: Path, text_only: bool) -> Path:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    stem = input_path.stem
    if stem.startswith("marker_dataset_"):
        stem = stem[len("marker_dataset_"):]
    mode = "textonly" if text_only else "chunks"
    model = os.environ.get("INFERENCE_MODEL", "llama3.2:3b").replace(":", "_").replace("/", "_")
    return EVIDENCE_RESULTS_DIR / f"baseline_{stem}_{mode}_{model}"


def main() -> None:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    args = parse_args()
    input_path = resolve_workspace_path(args.input_jsonl)
    if args.results_dir is None:
        results_dir = default_results_dir(input_path, args.text_only)
        print(f"results dir (auto): {results_dir}")
    else:
        results_dir = resolve_workspace_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="ia_cia_disc_like_",
        dir=results_dir,
        delete=False,
    ) as tmp:
        temp_disc_json = Path(tmp.name)
        json.dump({"DISC": load_records(input_path)}, tmp, ensure_ascii=False)

    cmd = [
        sys.executable,
        str(DISC_RUNNER),
        str(temp_disc_json),
        "--results-dir",
        str(results_dir),
    ]
    if args.text_only:
        cmd.append("--text-only")
    if args.text_chunks_only:
        cmd.append("--text-chunks-only")
    if args.max_docs is not None:
        cmd.extend(["--max-docs", str(args.max_docs)])
    cmd.extend(["--content-window", str(args.content_window)])
    if args.parallel_gpus:
        cmd.extend(["--parallel-gpus", str(args.parallel_gpus)])
    if args.gpu_ids:
        cmd.extend(["--gpu-ids", args.gpu_ids])

    env = os.environ.copy()
    env.setdefault("LLAMA_STACK_CONFIG", str(DEFAULT_LLAMA_STACK_CONFIG))
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="x00be x006a baseline_<x00ac>_<x01d3>_<x00ae>x0020 x01d4 x01d5")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--content-window", type=int, default=0,
                        help="x00ae x00af x01d6 x000d(x014a). main2 x0015 0=x0099 x009d. "
                             "(x00b1 x01d7 main1 x01d8 5000 x01d9)")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--text-chunks-only", action="store_true")
    parser.add_argument("--parallel-gpus", type=int, default=0)
    parser.add_argument("--gpu-ids", default="")
    args = parser.parse_args()
    if args.text_only and args.text_chunks_only:
        parser.error("--text-only and --text-chunks-only are mutually exclusive.")
    if not args.text_only and not args.text_chunks_only:
        args.text_only = True
    return args


def load_records(path: Path) -> list[dict[str, Any]]:
    """(Original documentation removed for the anonymized review release; see CODE_MAP.md.)"""
    records = []
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if not line.strip():
                continue
            source = json.loads(line)
            label = LABEL_MAP.get(str(source.get("label", "")).upper())
            if not label:
                raise ValueError(f"Unsupported label in record {index}: {source.get('label')}")
            text = source.get("marker_masked_text", "")
            records.append(
                {
                    "DocID": source.get("identifier", index),
                    "Title": f"IA CIA {source.get('identifier', index)}",
                    "Text": text,
                    "OCRtext": "",
                    "Classification": [{"Label": label}],
                    "SourceMetadata": {
                        "source": "ia_cia_readingroom",
                        "strict_high_conf_label": source.get("label", ""),
                        "removed_line_count": source.get("removed_line_count", ""),
                    },
                }
            )
    return records


if __name__ == "__main__":
    main()
