"""Cost-aware Taylor structured pruning for official YOLO11s on COCO2017.

The search never uses COCO val2017 to choose individual pruning actions. It
collects first-order Taylor channel scores on a fixed training subset, chooses
the lowest Taylor cost per actual GMAC saved, and validates only the baseline
and final pruned checkpoint on the full validation set.

Without --execute the script performs lineage, dataset, and model preflight only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.metadata
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
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules import C2f, C2PSA
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import init_seeds

from diagnose_coco2017_recovery import make_runtime_data_yaml
from prune_greedy_coco2017 import (
    Tee,
    backward_check,
    fingerprint,
    now,
    save_model,
    shape_signature,
    stats,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights/yolo11s.pt"
OFFICIAL_WEIGHT_SHA256 = (
    "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
)
UNSAFE_ROOTS = {"model.13.m.0.cv2.conv"}
TRIAL_FIELDS = [
    "step",
    "layer",
    "status",
    "reason",
    "before_channels",
    "after_channels",
    "removed_channels",
    "taylor_cost",
    "parameters",
    "gmacs",
    "step_gmac_saved",
    "total_compute_reduction",
    "score_per_gmac",
]


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checkpoint(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"Expected a trusted .pt checkpoint: {path}")
    return path


def append_csv(path: Path, row: dict[str, Any]) -> None:
    fresh = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        if fresh:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in TRIAL_FIELDS})


def candidate_layers(model: nn.Module) -> list[dict[str, Any]]:
    split_outputs = {
        module.cv1.conv
        for module in model.modules()
        if isinstance(module, (C2f, C2PSA))
    }
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module in split_outputs:
            continue
        if name in UNSAFE_ROOTS:
            continue
        if name == "model.0" or name.startswith(
            ("model.0.", "model.10.", "model.23.")
        ):
            continue
        if module.out_channels < 64:
            continue
        rows.append(
            {
                "layer": name,
                "initial_channels": module.out_channels,
                "kernel_size": list(module.kernel_size),
                "groups": module.groups,
            }
        )
    if not rows:
        raise RuntimeError("No eligible convolution roots were found")
    return rows


def collect_taylor_scores(
    base: Path,
    data_path: Path,
    candidates: list[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Collect mean per-image first-order |weight * gradient| channel scores."""
    init_seeds(args.seed, deterministic=True)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "rect": False})
    data = check_det_dataset(str(data_path), autodownload=False)
    dataset = build_yolo_dataset(
        cfg,
        data["train"],
        args.calibration_batch,
        data,
        mode="val",
        rect=False,
    )
    requested = min(args.calibration_images, len(dataset))
    generator = torch.Generator().manual_seed(args.seed)
    selected_indices = torch.randperm(len(dataset), generator=generator)[:requested].tolist()
    subset = torch.utils.data.Subset(dataset, selected_indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.calibration_batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    model = YOLO(str(base)).model.float().to(device)
    before = fingerprint(model)
    model.args = get_cfg(overrides=dict(model.args))
    model.criterion = None
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    modules = dict(model.named_modules())
    scores = {
        row["layer"]: torch.zeros(
            row["initial_channels"], dtype=torch.float64
        )
        for row in candidates
    }
    image_count = 0
    batch_losses: list[float] = []
    started = time.perf_counter()

    for batch_index, batch in enumerate(loader, start=1):
        batch = {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
        batch["img"] = batch["img"].float() / 255.0
        size = int(batch["img"].shape[0])
        model.zero_grad(set_to_none=True)
        loss, _ = model(batch)
        loss = loss.sum() / size
        if not torch.isfinite(loss).item():
            raise RuntimeError("Taylor calibration produced non-finite loss")
        loss.backward()

        for name, total in scores.items():
            weight = modules[name].weight
            if weight.grad is None or not torch.isfinite(weight.grad).all().item():
                raise RuntimeError(f"Missing or non-finite gradient for {name}")
            # Sum over each filter is the first-order Taylor channel cost.
            value = (weight.detach() * weight.grad.detach()).abs().flatten(1).sum(1)
            total.add_(value.double().cpu(), alpha=size)

        image_count += size
        batch_losses.append(float(loss.detach()))
        print(
            f"CALIBRATE {batch_index}/{len(loader)} "
            f"images={image_count}/{requested} loss={batch_losses[-1]:.6f}",
            flush=True,
        )

    if fingerprint(model) != before:
        raise RuntimeError("Taylor calibration changed model tensors or BN buffers")
    if image_count != requested:
        raise RuntimeError(
            f"Calibration used {image_count} images; expected {requested}"
        )

    values = {name: (total / image_count).tolist() for name, total in scores.items()}
    if not all(
        all(math.isfinite(value) for value in channel_scores)
        and max(channel_scores) > 0
        for channel_scores in values.values()
    ):
        raise RuntimeError("Taylor scores contain non-finite or all-zero values")

    summary = {
        "images": image_count,
        "batches": len(loader),
        "dataset_size": len(dataset),
        "selected_indices": selected_indices,
        "selection_sha256": hashlib.sha256(
            json.dumps(selected_indices).encode("utf-8")
        ).hexdigest(),
        "losses": batch_losses,
        "seconds": time.perf_counter() - started,
        "precision": "FP32",
        "bn_policy": "eval_no_updates",
        "checkpoint_unchanged": True,
        "score": "per-channel sum(abs(weight * gradient)), averaged by image",
    }
    del model, loader, subset, dataset, batch, loss, modules
    gc.collect()
    torch.cuda.empty_cache()
    return values, summary


def prune_action(
    model: nn.Module,
    name: str,
    indices: list[int],
    identities: dict[str, list[int]],
    example: torch.Tensor,
    initial: dict[str, int],
    min_remaining_ratio: float,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Apply one dependency-aware pruning action to a disposable model copy."""
    model.eval().float()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    modules = dict(model.named_modules())
    reverse = {module: module_name for module_name, module in modules.items()}
    root = modules.get(name)
    if not isinstance(root, nn.Conv2d):
        raise ValueError(f"Missing convolution root: {name}")

    split_outputs = {
        module.cv1.conv
        for module in model.modules()
        if isinstance(module, (C2f, C2PSA))
    }
    before = shape_signature(model)
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    group = graph.get_pruning_group(
        root, tp.prune_conv_out_channels, idxs=indices
    )
    if not graph.check_pruning_group(group):
        raise ValueError("dependency_group_rejected")

    output_deletions: dict[str, set[int]] = {}
    operations = []
    for dependency, dependent_indices in group:
        module = dependency.target.module
        module_name = reverse.get(module, "")
        out_pruning = graph.is_out_channel_pruning_fn(dependency.handler)

        if module in split_outputs and out_pruning:
            raise ValueError(f"protect_CSP_chunk_width:{module_name}")
        if module_name == "model.0" or module_name.startswith(
            ("model.0.", "model.10.", "model.23.dfl")
        ):
            raise ValueError(f"protect_stem_attention_DFL:{module_name}")

        if isinstance(module, nn.Conv2d):
            if (
                module_name.startswith("model.23.")
                and module_name.count(".") == 4
                and out_pruning
            ):
                raise ValueError(f"protect_detection_output:{module_name}")
            unique_indices = sorted(set(int(index) for index in dependent_indices))
            if out_pruning:
                output_deletions.setdefault(module_name, set()).update(unique_indices)
                minimum = max(
                    8, math.ceil(initial[module_name] * min_remaining_ratio)
                )
                remaining = module.out_channels - len(output_deletions[module_name])
                if remaining < minimum:
                    raise ValueError(
                        f"minimum_remaining_channels:{module_name}:{remaining}<{minimum}"
                    )
            operations.append(
                {
                    "layer": module_name,
                    "direction": "out" if out_pruning else "in",
                    "indices": unique_indices,
                }
            )

    updated = copy.deepcopy(identities)
    for module_name, removed in output_deletions.items():
        updated[module_name] = [
            original_id
            for current_id, original_id in enumerate(identities[module_name])
            if current_id not in removed
        ]

    before_channels = root.out_channels
    removed_original_ids = [identities[name][index] for index in indices]
    group.prune()

    for module_name, module in model.named_modules():
        if (
            isinstance(module, nn.Conv2d)
            and len(updated[module_name]) != module.out_channels
        ):
            raise ValueError(f"channel_identity_mismatch:{module_name}")

    after = shape_signature(model)
    changes = {
        key: {"before": before[key], "after": after[key]}
        for key in before
        if before[key] != after[key]
    }
    if not changes:
        raise ValueError("no_tensor_shape_change")

    for side in (320, args_imgsz(example)):
        with torch.no_grad():
            prediction = model(
                torch.zeros(1, 3, side, side, device=example.device)
            )[0]
        if prediction.shape[1] != 84 or not torch.isfinite(prediction).all().item():
            raise ValueError("invalid_detection_output")

    model.zero_grad(set_to_none=True)
    return updated, {
        "before_channels": before_channels,
        "after_channels": root.out_channels,
        "removed_channels": before_channels - root.out_channels,
        "removed_original_ids": removed_original_ids,
        "dependencies": operations,
        "changed_tensors": changes,
    }


def args_imgsz(example: torch.Tensor) -> int:
    return int(example.shape[-1])


def evaluate(
    weights: Path,
    data: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, float]:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        imgsz=args.imgsz,
        batch=args.val_batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        seed=args.seed,
        deterministic=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=False,
    )
    result = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "validation_inference_ms": float(metrics.speed["inference"]),
    }
    if not torch.isfinite(torch.tensor(list(result.values()))).all().item():
        raise RuntimeError("Validation produced non-finite metrics")
    del model, metrics
    torch.cuda.empty_cache()
    return result


def search(
    base: Path,
    candidates: list[dict[str, Any]],
    scores: dict[str, list[float]],
    device: torch.device,
    run_dir: Path,
    args: argparse.Namespace,
    info: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    model = YOLO(str(base)).model.float().to(device).eval()
    initial = {
        name: module.out_channels
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    identities = {name: list(range(width)) for name, width in initial.items()}
    example = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    baseline_params, baseline_gmacs = stats(model, example)
    current_gmacs = baseline_gmacs
    accepted_steps = []
    trials_path = run_dir / "candidates.csv"
    trials_path.unlink(missing_ok=True)
    stop_reason = "max_steps_reached"

    for step in range(1, args.max_steps + 1):
        best_trial = None
        for item in candidates:
            name = item["layer"]
            modules = dict(model.named_modules())
            root = modules.get(name)
            row: dict[str, Any] = {"step": step, "layer": name}
            if not isinstance(root, nn.Conv2d):
                row.update(status="unavailable", reason="root_removed_by_dependency")
                append_csv(trials_path, row)
                continue

            minimum = max(
                8, math.ceil(initial[name] * args.min_remaining_ratio)
            )
            removable = root.out_channels - minimum
            count = min(args.channel_step, removable)
            if count <= 0:
                row.update(status="unavailable", reason="minimum_width_reached")
                append_csv(trials_path, row)
                continue

            remaining_original_ids = identities[name]
            ranking = sorted(
                range(len(remaining_original_ids)),
                key=lambda current_id: (
                    scores[name][remaining_original_ids[current_id]],
                    remaining_original_ids[current_id],
                ),
            )
            indices = sorted(ranking[:count])
            taylor_cost = sum(
                scores[name][remaining_original_ids[index]] for index in indices
            )
            trial_model = copy.deepcopy(model)
            try:
                updated_identities, change = prune_action(
                    trial_model,
                    name,
                    indices,
                    identities,
                    example,
                    initial,
                    args.min_remaining_ratio,
                )
                parameters, gmacs = stats(trial_model, example)
                saved = current_gmacs - gmacs
                if saved <= 1e-9:
                    raise ValueError("no_positive_GMAC_saving")
                score = taylor_cost / saved
                total_reduction = 1.0 - gmacs / baseline_gmacs
                row.update(
                    status="feasible",
                    reason="",
                    taylor_cost=taylor_cost,
                    parameters=parameters,
                    gmacs=gmacs,
                    step_gmac_saved=saved,
                    total_compute_reduction=total_reduction,
                    score_per_gmac=score,
                    **{
                        key: change[key]
                        for key in (
                            "before_channels",
                            "after_channels",
                            "removed_channels",
                        )
                    },
                )
                candidate = {
                    "row": row,
                    "model": trial_model,
                    "identities": updated_identities,
                    "change": change,
                }
                if (
                    best_trial is None
                    or row["score_per_gmac"]
                    < best_trial["row"]["score_per_gmac"]
                ):
                    if best_trial is not None:
                        del best_trial["model"]
                    best_trial = candidate
                else:
                    del trial_model
            except (
                ValueError,
                RuntimeError,
                IndexError,
                KeyError,
                AssertionError,
            ) as error:
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    raise
                row.update(
                    status="rejected",
                    reason=f"{type(error).__name__}: {error}",
                )
                del trial_model
            append_csv(trials_path, row)
            gc.collect()

        if best_trial is None:
            stop_reason = "no_feasible_actions"
            break

        chosen = best_trial["row"]
        chosen["status"] = "accepted"
        chosen["details"] = best_trial["change"]
        model = best_trial["model"]
        identities = best_trial["identities"]
        current_gmacs = chosen["gmacs"]
        accepted_steps.append(chosen)
        info["steps"] = accepted_steps
        info["status"] = "pruning"
        write_json(run_dir / "steps.json", accepted_steps)
        save_model(model, run_dir / "pruned_partial.pt", base)
        write_json(run_dir / "run_info.json", info)
        print(
            f"ACCEPT step={step} layer={chosen['layer']} "
            f"{chosen['before_channels']}->{chosen['after_channels']} "
            f"GMAC_reduction={chosen['total_compute_reduction']:.4%}",
            flush=True,
        )
        if chosen["total_compute_reduction"] >= args.target_reduction:
            stop_reason = "target_reached"
            break
        gc.collect()
        torch.cuda.empty_cache()

    if not accepted_steps:
        raise RuntimeError("Taylor search found no feasible pruning action")

    backward_check(model, device)
    final_path = run_dir / "pruned_raw.pt"
    save_model(model, final_path, base)
    reloaded = YOLO(str(final_path)).model
    if shape_signature(reloaded) != shape_signature(model):
        raise RuntimeError("Saved Taylor-pruned architecture failed reload check")
    final_params, final_gmacs = stats(model, example)
    result = {
        "stop_reason": stop_reason,
        "parameters": final_params,
        "gmacs": final_gmacs,
        "parameter_reduction": 1.0 - final_params / baseline_params,
        "compute_reduction": 1.0 - final_gmacs / baseline_gmacs,
        "baseline_parameters": baseline_params,
        "baseline_gmacs": baseline_gmacs,
        "checkpoint": str(final_path),
        "checkpoint_sha256": sha256(final_path),
        "reload_verified": True,
    }
    del model, reloaded, example
    gc.collect()
    torch.cuda.empty_cache()
    return final_path, result


def write_report(run_dir: Path, info: dict[str, Any]) -> None:
    baseline = info["baseline"]
    raw = info["raw"]
    search_result = info["search_result"]
    lines = [
        "# Experiment 15: Taylor structured pruning on COCO2017",
        "",
        "The parent is the official YOLO11s checkpoint recorded by SHA256.",
        "Pruning actions were selected on a fixed training subset without using val2017.",
        "",
        "| Metric | Parent | Taylor-pruned raw |",
        "|---|---:|---:|",
        f"| Parameters | {search_result['baseline_parameters']:,} | {search_result['parameters']:,} |",
        f"| GMACs | {search_result['baseline_gmacs']:.4f} | {search_result['gmacs']:.4f} |",
        f"| GMAC reduction | 0.00% | {search_result['compute_reduction']:.2%} |",
        f"| mAP50-95 | {baseline['map50_95']:.4f} | {raw['map50_95']:.4f} |",
        "",
        f"Stop reason: `{search_result['stop_reason']}`.",
        f"Raw mAP50-95 change: {raw['map50_95'] - baseline['map50_95']:+.4f}.",
        "",
        "Recovery training is deliberately separate. Run the recovery diagnostic "
        "with this run_info.json as provenance before choosing a deployable model.",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace, run_dir: Path, info: dict[str, Any]) -> None:
    base = checkpoint(args.weights)
    actual_weight_hash = sha256(base)
    if (
        not args.allow_weight_hash_mismatch
        and actual_weight_hash != OFFICIAL_WEIGHT_SHA256
    ):
        raise RuntimeError(
            "YOLO11s baseline hash mismatch. "
            f"Expected {OFFICIAL_WEIGHT_SHA256}, got {actual_weight_hash}."
        )

    data_info = make_runtime_data_yaml(
        args.data,
        args.dataset_root,
        run_dir / "coco2017_runtime.yaml",
    )
    cpu_model = YOLO(str(base)).model
    candidates = candidate_layers(cpu_model)
    parameter_count = sum(parameter.numel() for parameter in cpu_model.parameters())
    del cpu_model

    info.update(
        {
            "status": "preflight_complete",
            "configuration": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "sources": {
                str(base): actual_weight_hash,
                str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
                str(Path(args.data).resolve()): sha256(Path(args.data).resolve()),
            },
            "data": data_info,
            "parent": {
                "path": str(base),
                "sha256": actual_weight_hash,
                "parameters": parameter_count,
            },
            "candidate_roots": candidates,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": importlib.metadata.version("torch"),
                "ultralytics": importlib.metadata.version("ultralytics"),
                "torch_pruning": importlib.metadata.version("torch-pruning"),
                "cuda_available": torch.cuda.is_available(),
            },
        }
    )
    write_json(run_dir / "run_info.json", info)
    print(
        f"PREFLIGHT_COMPLETE parent={actual_weight_hash} "
        f"candidates={len(candidates)}",
        flush=True,
    )

    if not args.execute:
        info["elapsed_seconds"] = time.perf_counter() - info.pop("_timer")
        write_json(run_dir / "run_info.json", info)
        print(
            "No calibration, validation, or pruning was run. "
            "Pass --execute on the GPU server.",
            flush=True,
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Taylor pruning requires a CUDA GPU")
    if not str(args.device).isdigit():
        raise ValueError("--device must be a single CUDA index")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    info["environment"]["gpu"] = torch.cuda.get_device_name(device)
    info["status"] = "calibrating"
    write_json(run_dir / "run_info.json", info)

    runtime_data = Path(data_info["runtime"])
    scores, calibration = collect_taylor_scores(
        base, runtime_data, candidates, device, args
    )
    info["calibration"] = calibration
    torch.save(scores, run_dir / "taylor_scores.pt")
    info["taylor_scores_sha256"] = sha256(run_dir / "taylor_scores.pt")
    info["status"] = "validating_baseline"
    write_json(run_dir / "run_info.json", info)

    baseline = evaluate(
        base,
        runtime_data,
        run_dir / "validation/baseline",
        args,
    )
    info["baseline"] = baseline
    info["status"] = "pruning"
    info["steps"] = []
    write_json(run_dir / "run_info.json", info)

    pruned, search_result = search(
        base, candidates, scores, device, run_dir, args, info
    )
    info["search_result"] = search_result
    info["status"] = "validating_pruned"
    write_json(run_dir / "run_info.json", info)

    raw = evaluate(
        pruned,
        runtime_data,
        run_dir / "validation/pruned_raw",
        args,
    )
    info["raw"] = raw
    info.update(
        {
            "status": "complete",
            "finished_at": now(),
            "elapsed_seconds": time.perf_counter() - info.pop("_timer"),
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1e6,
        }
    )
    write_report(run_dir, info)
    write_json(run_dir / "run_info.json", info)
    print(f"COMPLETE {run_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "configs/coco2017.yaml"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=2048)
    parser.add_argument("--calibration-batch", type=int, default=16)
    parser.add_argument("--val-batch", type=int, default=64)
    parser.add_argument("--channel-step", type=int, default=8)
    parser.add_argument("--target-reduction", type=float, default=0.10)
    parser.add_argument("--min-remaining-ratio", type=float, default=0.50)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-weight-hash-mismatch",
        action="store_true",
        help="Use only for an explicitly documented alternative baseline.",
    )
    args = parser.parse_args()
    if args.calibration_images < 1 or args.calibration_batch < 1:
        parser.error("Calibration image count and batch must be positive")
    if args.val_batch < 1 or args.channel_step < 1 or args.max_steps < 1:
        parser.error("Validation batch, channel step, and max steps must be positive")
    if not 0 < args.target_reduction < 1:
        parser.error("target-reduction must be between 0 and 1")
    if not 0 < args.min_remaining_ratio < 1:
        parser.error("min-remaining-ratio must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    run_dir = (
        ROOT
        / "runs/prune/experiment15_taylor_coco2017"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    info: dict[str, Any] = {
        "started_at": now(),
        "status": "starting",
        "_timer": time.perf_counter(),
    }
    original_out, original_err = sys.stdout, sys.stderr
    with (run_dir / "run.log").open("w", encoding="utf-8", buffering=1) as logfile:
        sys.stdout = Tee(original_out, logfile)
        sys.stderr = Tee(original_err, logfile)
        handlers = [
            (handler, handler.stream)
            for handler in LOGGER.handlers
            if hasattr(handler, "stream")
        ]
        for handler, _ in handlers:
            handler.setStream(sys.stdout)
        try:
            print(f"RUN_DIR {run_dir}", flush=True)
            run(args, run_dir, info)
        except BaseException as error:
            info.update(
                {
                    "status": (
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed"
                    ),
                    "finished_at": now(),
                    "elapsed_seconds": time.perf_counter() - info.pop("_timer"),
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )
            write_json(run_dir / "run_info.json", info)
            traceback.print_exc()
            raise
        finally:
            for handler, stream in handlers:
                handler.setStream(stream)
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == "__main__":
    main()

