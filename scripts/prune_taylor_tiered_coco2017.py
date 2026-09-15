"""Tiered Taylor/GMAC greedy pruning for official YOLO11s on COCO2017.

Layer tiers come from an independent train-derived sensitivity scan.  Each
accepted action recalibrates Taylor scores on the fixed calibration split.
COCO val2017 is used only for the final endpoint measurement.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

from prune_greedy_coco2017 import Tee, backward_check, fingerprint, save_model, shape_signature, stats, write_json
from prune_taylor_coco2017 import (
    OFFICIAL_WEIGHT_SHA256,
    candidate_layers,
    checkpoint,
    collect_taylor_scores,
    evaluate,
    make_runtime_data_yaml,
    prune_action,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "step", "layer", "tier", "status", "reason", "before_channels",
    "after_channels", "removed_channels", "layer_removed_total", "layer_cap",
    "taylor_cost", "parameters", "gmacs", "step_gmac_saved",
    "total_compute_reduction", "score_per_gmac",
]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def load_sensitivity(run_dir: Path, expected_sha: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    info_path, csv_path = run_dir / "run_info.json", run_dir / "sensitivity.csv"
    if not info_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(f"Incomplete sensitivity run: {run_dir}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("status") != "complete":
        raise RuntimeError("Sensitivity run is not complete")
    if info.get("source", {}).get("sha256") != expected_sha:
        raise RuntimeError("Sensitivity scan used a different source checkpoint")
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                row["map_drop_per_gmac"] = float(row["map_drop_per_gmac"])
                rows.append(row)
    if not rows:
        raise RuntimeError("No feasible sensitivity rows")
    rows.sort(key=lambda row: (row["map_drop_per_gmac"], row["layer"]))
    return rows, info


def assign_tiers(rows: list[dict[str, Any]], low_fraction: float, medium_fraction: float) -> dict[str, str]:
    count = len(rows)
    low_end = max(1, math.ceil(count * low_fraction))
    medium_end = min(count, low_end + math.ceil(count * medium_fraction))
    return {
        row["layer"]: ("low" if index < low_end else "medium" if index < medium_end else "high")
        for index, row in enumerate(rows)
    }


def cap_channels(width: int, fraction: float, step: int) -> int:
    return int(math.floor(width * fraction / step) * step)


def append_row(path: Path, row: dict[str, Any]) -> None:
    fresh = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDS})


def verify_caps(model: nn.Module, initial: dict[str, int], tiers: dict[str, str], caps: dict[str, int]) -> None:
    modules = dict(model.named_modules())
    for name, tier in tiers.items():
        module = modules.get(name)
        if not isinstance(module, nn.Conv2d):
            raise ValueError(f"tier_root_missing:{name}")
        removed = initial[name] - module.out_channels
        if removed > caps[name]:
            raise ValueError(f"tier_cap_exceeded:{name}:{removed}>{caps[name]}:{tier}")


def search(base: Path, calibration_yaml: Path, sensitivity_rows: list[dict[str, Any]],
           tiers: dict[str, str], device: torch.device, run_dir: Path,
           args: argparse.Namespace, info: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    model = YOLO(str(base)).model.float().to(device).eval()
    initial = {name: module.out_channels for name, module in model.named_modules() if isinstance(module, nn.Conv2d)}
    identities = {name: list(range(width)) for name, width in initial.items()}
    example = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    base_params, base_gmacs = stats(model, example)
    current_gmacs = base_gmacs
    caps = {
        name: cap_channels(initial[name], args.low_cap if tier == "low" else args.medium_cap if tier == "medium" else 0.0, args.channel_step)
        for name, tier in tiers.items()
    }
    accepted: list[dict[str, Any]] = []
    trials_path = run_dir / "candidates.csv"
    stop_reason = "max_steps_reached"

    for step in range(1, args.max_steps + 1):
        active = []
        modules = dict(model.named_modules())
        for name, tier in tiers.items():
            module = modules.get(name)
            if tier != "high" and isinstance(module, nn.Conv2d) and initial[name] - module.out_channels < caps[name]:
                active.append({"layer": name, "initial_channels": module.out_channels})
        if not active:
            stop_reason = "all_tier_caps_reached"
            break

        score_source = run_dir / f"taylor_step_{step:02d}.pt"
        save_model(model, score_source, base)
        scores, score_summary = collect_taylor_scores(score_source, calibration_yaml, active, device, args)
        write_json(run_dir / f"taylor_step_{step:02d}.json", score_summary)
        best = None
        for item in active:
            name, tier = item["layer"], tiers[item["layer"]]
            root = dict(model.named_modules()).get(name)
            row: dict[str, Any] = {"step": step, "layer": name, "tier": tier, "layer_cap": caps[name]}
            removed_total = initial[name] - root.out_channels
            count = min(args.channel_step, caps[name] - removed_total)
            if count <= 0:
                row.update(status="unavailable", reason="tier_cap_reached")
                append_row(trials_path, row)
                continue
            ranking = sorted(range(root.out_channels), key=lambda i: (scores[name][i], i))
            indices = sorted(ranking[:count])
            taylor_cost = sum(scores[name][index] for index in indices)
            trial = copy.deepcopy(model)
            try:
                new_ids, change = prune_action(trial, name, indices, identities, example, initial, args.min_remaining_ratio)
                verify_caps(trial, initial, tiers, caps)
                parameters, gmacs = stats(trial, example)
                saved = current_gmacs - gmacs
                if saved <= 1e-9:
                    raise ValueError("no_positive_GMAC_saving")
                reduction = 1.0 - gmacs / base_gmacs
                row.update(
                    status="feasible", reason="", taylor_cost=taylor_cost,
                    parameters=parameters, gmacs=gmacs, step_gmac_saved=saved,
                    total_compute_reduction=reduction, score_per_gmac=taylor_cost / saved,
                    layer_removed_total=initial[name] - change["after_channels"],
                    **{key: change[key] for key in ("before_channels", "after_channels", "removed_channels")},
                )
                candidate = {"row": row, "model": trial, "identities": new_ids, "change": change}
                if best is None or (row["score_per_gmac"], -saved, name) < (best["row"]["score_per_gmac"], -best["row"]["step_gmac_saved"], best["row"]["layer"]):
                    if best is not None:
                        del best["model"]
                    best = candidate
                else:
                    del trial
            except (ValueError, RuntimeError, IndexError, KeyError, AssertionError) as error:
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    raise
                row.update(status="rejected", reason=f"{type(error).__name__}: {error}")
                del trial
            append_row(trials_path, row)
            gc.collect()

        score_source.unlink(missing_ok=True)
        if best is None:
            stop_reason = "no_feasible_actions"
            break
        chosen = best["row"]
        chosen["status"] = "accepted"
        chosen["details"] = best["change"]
        model, identities, current_gmacs = best["model"], best["identities"], chosen["gmacs"]
        accepted.append(chosen)
        info.update(status="pruning", steps=accepted)
        write_json(run_dir / "steps.json", accepted)
        save_model(model, run_dir / "pruned_partial.pt", base)
        write_json(run_dir / "run_info.json", info)
        print(f"ACCEPT step={step} layer={chosen['layer']} tier={chosen['tier']} "
              f"{chosen['before_channels']}->{chosen['after_channels']} "
              f"GMAC_reduction={chosen['total_compute_reduction']:.4%}", flush=True)
        if chosen["total_compute_reduction"] >= args.target_reduction:
            stop_reason = "target_reached"
            break
        gc.collect()
        torch.cuda.empty_cache()

    if not accepted:
        raise RuntimeError("Tiered Taylor search found no feasible action")
    backward_check(model, device)
    output = run_dir / "pruned_raw.pt"
    save_model(model, output, base)
    reloaded = YOLO(str(output)).model
    if shape_signature(reloaded) != shape_signature(model):
        raise RuntimeError("Saved architecture failed reload verification")
    parameters, gmacs = stats(model, example)
    result = {
        "stop_reason": stop_reason, "accepted_steps": len(accepted),
        "baseline_parameters": base_params, "parameters": parameters,
        "parameter_reduction": 1.0 - parameters / base_params,
        "baseline_gmacs": base_gmacs, "gmacs": gmacs,
        "compute_reduction": 1.0 - gmacs / base_gmacs,
        "checkpoint": str(output), "checkpoint_sha256": sha256(output),
        "tier_caps_channels": caps, "reload_verified": True,
    }
    del model, reloaded, example
    gc.collect()
    torch.cuda.empty_cache()
    return output, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/yolo11s.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/coco2017.yaml")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sensitivity-run", type=Path, required=True)
    parser.add_argument("--target-reduction", type=float, default=0.05)
    parser.add_argument("--low-fraction", type=float, default=0.40)
    parser.add_argument("--medium-fraction", type=float, default=0.35)
    parser.add_argument("--low-cap", type=float, default=0.125)
    parser.add_argument("--medium-cap", type=float, default=0.0625)
    parser.add_argument("--channel-step", type=int, default=8)
    parser.add_argument("--min-remaining-ratio", type=float, default=0.50)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--calibration-images", type=int, default=2048)
    parser.add_argument("--calibration-batch", type=int, default=32)
    parser.add_argument("--val-batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 0 < args.target_reduction < 1 or args.channel_step < 1:
        parser.error("Invalid target reduction or channel step")
    if args.low_fraction <= 0 or args.medium_fraction < 0 or args.low_fraction + args.medium_fraction > 1:
        parser.error("Invalid tier fractions")
    return args


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs/prune/experiment17_tiered_taylor_coco2017" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    info: dict[str, Any] = {"started_at": now(), "status": "starting", "script": str(Path(__file__).resolve())}
    log_handle = (run_dir / "run.log").open("a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(old_out, log_handle), Tee(old_err, log_handle)
    try:
        base = checkpoint(args.weights)
        source_sha = sha256(base)
        if source_sha != OFFICIAL_WEIGHT_SHA256:
            raise RuntimeError(f"Official checkpoint hash mismatch: {source_sha}")
        sensitivity_rows, sensitivity_info = load_sensitivity(args.sensitivity_run, source_sha)
        tiers = assign_tiers(sensitivity_rows, args.low_fraction, args.medium_fraction)
        calibration_yaml = Path(sensitivity_info["splits"]["calibration_yaml"])
        if not calibration_yaml.is_file():
            raise FileNotFoundError(calibration_yaml)
        full_data = run_dir / "coco2017_runtime.yaml"
        data_info = make_runtime_data_yaml(args.data, args.dataset_root, full_data)
        counts = {tier: sum(value == tier for value in tiers.values()) for tier in ("low", "medium", "high")}
        info.update(
            status="preflight_complete", source={"path": str(base), "sha256": source_sha},
            sensitivity={"run": str(args.sensitivity_run.resolve()), "rows": len(sensitivity_rows), "tiers": counts},
            configuration={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            data=data_info,
        )
        write_json(run_dir / "tiers.json", {"tiers": tiers, "ranked_sensitivity": sensitivity_rows})
        write_json(run_dir / "run_info.json", info)
        print(f"PREFLIGHT_COMPLETE tiers={counts}", flush=True)
        if not args.execute:
            return 0
        device = torch.device("cuda:0" if str(args.device) == "0" and torch.cuda.is_available() else args.device)
        if device.type != "cuda":
            raise RuntimeError("Formal run requires CUDA")
        info["status"] = "pruning"
        write_json(run_dir / "run_info.json", info)
        raw_path, result = search(base, calibration_yaml, sensitivity_rows, tiers, device, run_dir, args, info)
        info["search_result"] = result
        info["status"] = "validating"
        write_json(run_dir / "run_info.json", info)
        info["raw_validation"] = evaluate(raw_path, full_data, run_dir / "validation_raw", args)
        info.update(status="complete", finished_at=now())
        write_json(run_dir / "run_info.json", info)
        (run_dir / "report.md").write_text(
            "# Experiment 17: tiered Taylor pruning\n\n"
            f"- GMAC reduction: {result['compute_reduction']:.2%}\n"
            f"- Parameter reduction: {result['parameter_reduction']:.2%}\n"
            f"- Raw COCO val mAP50-95: {info['raw_validation']['map50_95']:.4f}\n"
            f"- Stop reason: `{result['stop_reason']}`\n",
            encoding="utf-8",
        )
        print(f"RUN_COMPLETE {run_dir}", flush=True)
        return 0
    except Exception as error:
        info.update(status="failed", finished_at=now(), error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        write_json(run_dir / "run_info.json", info)
        traceback.print_exc()
        return 1
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
