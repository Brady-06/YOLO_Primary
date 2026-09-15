"""Measure dependency-aware Taylor sensitivity on safe YOLO11s roots.

The official checkpoint is tested layer by layer from the same starting state.
Taylor gradients use a fixed train2017 calibration split; candidate mAP uses a
disjoint train2017 tuning split. COCO val2017 is not used for layer selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO

from prune_greedy_coco2017 import Tee, now, save_model, stats, write_json
from prune_taylor_coco2017 import (
    OFFICIAL_WEIGHT_SHA256,
    candidate_layers,
    checkpoint,
    collect_taylor_scores,
    evaluate,
    prune_action,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "rank",
    "layer",
    "status",
    "reason",
    "initial_channels",
    "removed_channels",
    "taylor_cost",
    "parameters",
    "gmacs",
    "gmac_saved",
    "gmac_reduction",
    "baseline_map50_95",
    "trial_map50_95",
    "map50_95_drop",
    "map_drop_per_gmac",
]


def absolute_image(dataset_root: Path, value: str) -> str:
    value = value.strip()
    path = Path(value)
    if path.is_absolute():
        return path.as_posix()
    if value.startswith("./"):
        value = value[2:]
    return (dataset_root / value).resolve().as_posix()


def prepare_splits(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    source = args.data.expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    source_value = Path(str(payload["train"]))
    source_path = source_value if source_value.is_absolute() else dataset_root / source_value
    if source_path.is_file():
        lines = [
            absolute_image(dataset_root, line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif source_path.is_dir():
        image_suffixes = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
        lines = sorted(
            path.resolve().as_posix()
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in image_suffixes
        )
    else:
        raise FileNotFoundError(f"COCO train source does not exist: {source_path}")
    if len(lines) != 118287:
        raise RuntimeError(f"Expected 118287 train2017 images, got {len(lines)}")
    if args.calibration_images + args.tune_images >= len(lines):
        raise ValueError("Calibration and tuning splits leave no recovery images")

    order = torch.randperm(
        len(lines), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    calibration = [lines[i] for i in order[: args.calibration_images]]
    tune = [
        lines[i]
        for i in order[
            args.calibration_images : args.calibration_images + args.tune_images
        ]
    ]
    recovery = [lines[i] for i in order[args.calibration_images + args.tune_images :]]

    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = split_dir / "source_manifest.txt"
    source_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths = {}
    for name, values in (
        ("calibration", calibration),
        ("tune", tune),
        ("recovery_train", recovery),
    ):
        path = split_dir / f"{name}.txt"
        path.write_text("\n".join(values) + "\n", encoding="utf-8")
        paths[name] = path

    base_yaml = {
        key: value
        for key, value in payload.items()
        if key not in {"download", "path", "train", "val", "test"}
    }
    calibration_yaml = run_dir / "calibration.yaml"
    tune_yaml = run_dir / "tune.yaml"
    recovery_yaml = run_dir / "recovery.yaml"
    for destination, train_value, val_value in (
        (calibration_yaml, paths["calibration"], paths["tune"]),
        (tune_yaml, paths["recovery_train"], paths["tune"]),
        (recovery_yaml, paths["recovery_train"], paths["tune"]),
    ):
        value = {
            "path": dataset_root.as_posix(),
            "train": train_value.as_posix(),
            "val": val_value.as_posix(),
            **base_yaml,
        }
        destination.write_text(
            yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return {
        "dataset_root": str(dataset_root),
        "source": str(source_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "counts": {
            "calibration": len(calibration),
            "tune": len(tune),
            "recovery_train": len(recovery),
        },
        "files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "calibration_yaml": str(calibration_yaml),
        "tune_yaml": str(tune_yaml),
        "recovery_yaml": str(recovery_yaml),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)


def run(args: argparse.Namespace, run_dir: Path, info: dict[str, Any]) -> None:
    base = checkpoint(args.weights)
    actual_hash = sha256(base)
    if actual_hash != OFFICIAL_WEIGHT_SHA256:
        raise RuntimeError(
            f"Expected official YOLO11s {OFFICIAL_WEIGHT_SHA256}, got {actual_hash}"
        )
    split_info = prepare_splits(args, run_dir)
    cpu_model = YOLO(str(base)).model
    candidates = candidate_layers(cpu_model)
    del cpu_model
    info.update(
        status="preflight_complete",
        configuration={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        source={"path": str(base), "sha256": actual_hash},
        splits=split_info,
        candidate_count=len(candidates),
        candidates=candidates,
        environment={"python": sys.version, "platform": platform.platform()},
    )
    write_json(run_dir / "run_info.json", info)
    print(f"PREFLIGHT_COMPLETE candidates={len(candidates)}", flush=True)
    if not args.execute:
        return
    if not torch.cuda.is_available() or not str(args.device).isdigit():
        raise RuntimeError("Execution requires one CUDA device")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    scores, calibration = collect_taylor_scores(
        base,
        Path(split_info["calibration_yaml"]),
        candidates,
        device,
        args,
    )
    torch.save(scores, run_dir / "taylor_scores.pt")
    info.update(status="measuring_sensitivity", calibration=calibration)
    write_json(run_dir / "run_info.json", info)

    reference = YOLO(str(base)).model.float().to(device).eval()
    initial = {
        name: module.out_channels
        for name, module in reference.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    example = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    baseline_params, baseline_gmacs = stats(reference, example)
    del reference
    baseline = evaluate(
        base,
        Path(split_info["tune_yaml"]),
        run_dir / "validation/baseline_tune",
        args,
    )
    rows: list[dict[str, Any]] = []
    temporary = run_dir / "candidate.pt"

    for index, item in enumerate(candidates, start=1):
        name = item["layer"]
        row: dict[str, Any] = {
            "rank": index,
            "layer": name,
            "status": "rejected",
            "reason": "",
            "initial_channels": initial[name],
            "baseline_map50_95": baseline["map50_95"],
        }
        trial = YOLO(str(base)).model.float().to(device).eval()
        identities = {
            module_name: list(range(width))
            for module_name, width in initial.items()
        }
        try:
            cap = max(8, int(math.floor(initial[name] * args.layer_cap / 8) * 8))
            count = min(args.channel_step, cap, initial[name] - 8)
            if count <= 0:
                raise ValueError("no_removable_channels")
            ranking = sorted(range(initial[name]), key=lambda i: (scores[name][i], i))
            indices = sorted(ranking[:count])
            taylor_cost = sum(scores[name][i] for i in indices)
            _, change = prune_action(
                trial,
                name,
                indices,
                identities,
                example,
                initial,
                args.min_remaining_ratio,
            )
            parameters, gmacs = stats(trial, example)
            saved = baseline_gmacs - gmacs
            if saved <= 1e-9:
                raise ValueError("no_positive_GMAC_saving")
            save_model(trial, temporary, base)
            measured = evaluate(
                temporary,
                Path(split_info["tune_yaml"]),
                run_dir / "validation" / f"candidate_{index:02d}",
                args,
            )
            drop = baseline["map50_95"] - measured["map50_95"]
            row.update(
                status="ok",
                removed_channels=change["removed_channels"],
                taylor_cost=taylor_cost,
                parameters=parameters,
                gmacs=gmacs,
                gmac_saved=saved,
                gmac_reduction=1.0 - gmacs / baseline_gmacs,
                trial_map50_95=measured["map50_95"],
                map50_95_drop=drop,
                map_drop_per_gmac=drop / saved,
            )
        except (ValueError, RuntimeError, KeyError, IndexError, AssertionError) as error:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            row["reason"] = f"{type(error).__name__}: {error}"
        finally:
            temporary.unlink(missing_ok=True)
            del trial
            torch.cuda.empty_cache()
        rows.append(row)
        write_csv(run_dir / "sensitivity.csv", rows)
        info["results"] = rows
        write_json(run_dir / "run_info.json", info)
        print(
            f"SENSITIVITY {index}/{len(candidates)} {name} {row['status']} "
            f"drop={row.get('map50_95_drop', '')}",
            flush=True,
        )

    feasible = [row for row in rows if row["status"] == "ok"]
    feasible.sort(key=lambda row: (row["map_drop_per_gmac"], row["layer"]))
    for rank, row in enumerate(feasible, start=1):
        row["rank"] = rank
    rejected = [row for row in rows if row["status"] != "ok"]
    write_csv(run_dir / "sensitivity.csv", feasible + rejected)
    info.update(
        status="complete",
        finished_at=now(),
        baseline={
            **baseline,
            "parameters": baseline_params,
            "gmacs": baseline_gmacs,
        },
        results=feasible + rejected,
        elapsed_seconds=time.perf_counter() - info.pop("_timer"),
    )
    write_json(run_dir / "run_info.json", info)
    print(f"COMPLETE {run_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/yolo11s.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/coco2017.yaml")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=2048)
    parser.add_argument("--tune-images", type=int, default=2048)
    parser.add_argument("--calibration-batch", type=int, default=32)
    parser.add_argument("--val-batch", type=int, default=128)
    parser.add_argument("--channel-step", type=int, default=8)
    parser.add_argument("--layer-cap", type=float, default=0.125)
    parser.add_argument("--min-remaining-ratio", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if min(
        args.calibration_images,
        args.tune_images,
        args.calibration_batch,
        args.val_batch,
        args.channel_step,
    ) < 1:
        parser.error("Image counts, batches, and channel step must be positive")
    if not 0 < args.layer_cap <= 0.5 or not 0 < args.min_remaining_ratio < 1:
        parser.error("Invalid channel ratio")
    return args


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    run_dir = (
        ROOT
        / "runs/analysis/experiment16_taylor_sensitivity_coco2017"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    info: dict[str, Any] = {
        "started_at": now(),
        "status": "starting",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "_timer": time.perf_counter(),
    }
    try:
        with (run_dir / "run.log").open("w", encoding="utf-8", buffering=1) as log:
            original_out, original_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_err, log)
            try:
                run(args, run_dir, info)
            finally:
                sys.stdout, sys.stderr = original_out, original_err
    except BaseException as error:
        info.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            finished_at=now(),
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        if "_timer" in info:
            info["elapsed_seconds"] = time.perf_counter() - info.pop("_timer")
        write_json(run_dir / "run_info.json", info)
        raise


if __name__ == "__main__":
    main()
