"""Continue the verified 5% checkpoint to nested 10/15/20% Taylor stages."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

from diagnose_coco2017_recovery import evaluate, make_runtime_data_yaml, sha256, write_json
from prune_greedy_coco2017 import Tee, backward_check, save_model, shape_signature, stats
from prune_taylor_coco2017 import OFFICIAL_WEIGHT_SHA256, collect_taylor_scores, prune_action
from prune_taylor_tiered_coco2017 import assign_tiers, load_sensitivity
from recalibrate_bn_coco2017 import recalibrate, subset_yaml

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["stage", "step", "layer", "tier", "status", "reason", "before_channels",
          "after_channels", "removed_channels", "layer_removed_total", "layer_cap",
          "taylor_cost", "gmacs", "step_gmac_saved", "total_compute_reduction", "score_per_gmac"]
CAPS = {
    0.10: {"low": 0.25, "medium": 0.125, "high": 0.0625},
    0.15: {"low": 0.375, "medium": 0.25, "high": 0.125},
    0.20: {"low": 0.50, "medium": 0.375, "high": 0.25},
}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def append_row(path: Path, row: dict[str, Any]) -> None:
    fresh = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDS})


def stage_caps(initial: dict[str, int], tiers: dict[str, str], target: float, step: int) -> dict[str, int]:
    fractions = CAPS[target]
    return {name: int(math.floor(initial[name] * fractions[tier] / step) * step) for name, tier in tiers.items()}


def verify_caps(model: nn.Module, initial: dict[str, int], caps: dict[str, int]) -> None:
    modules = dict(model.named_modules())
    for name, cap in caps.items():
        module = modules.get(name)
        if not isinstance(module, nn.Conv2d):
            raise ValueError(f"tier_root_missing:{name}")
        removed = initial[name] - module.out_channels
        if removed > cap:
            raise ValueError(f"stage_cap_exceeded:{name}:{removed}>{cap}")


def endpoint(model: nn.Module, template: Path, target: float, run_dir: Path,
             calibration8: Path, tune: Path, full: Path, args, eval_args) -> dict[str, Any]:
    label = f"stage_{int(target * 100):02d}"
    raw = run_dir / label / "raw.pt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, raw, template)
    raw_tune = evaluate(raw, tune, run_dir / label / "validation_tune_raw", eval_args)
    bn = run_dir / label / "bn_m0005_8192.pt"
    bn_details = recalibrate(raw, calibration8, bn, 8192, args.bn_batch, args.workers,
                             torch.device("cuda:0"), args.seed, 0.005, False)
    bn_tune = evaluate(bn, tune, run_dir / label / "validation_tune_bn", eval_args)
    selected = bn if bn_tune["map50_95"] >= raw_tune["map50_95"] else raw
    selected_label = "bn_m0005_8192" if selected == bn else "raw"
    full_metrics = evaluate(selected, full, run_dir / label / "validation_full_selected", eval_args)
    return {"target": target, "raw": {"path": str(raw), "sha256": sha256(raw), "tune": raw_tune},
            "bn": {"path": str(bn), "sha256": sha256(bn), "tune": bn_tune, "details": bn_details},
            "selected": {"label": selected_label, "path": str(selected), "sha256": sha256(selected),
                         "tune": bn_tune if selected == bn else raw_tune, "full_val": full_metrics}}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=ROOT / "weights/yolo11s.pt")
    p.add_argument("--stage5-run", type=Path, required=True)
    p.add_argument("--sensitivity-run", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "configs/coco2017.yaml")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--calibration-images", type=int, default=2048)
    p.add_argument("--calibration-batch", type=int, default=32)
    p.add_argument("--bn-batch", type=int, default=128)
    p.add_argument("--val-batch", type=int, default=128)
    p.add_argument("--channel-step", type=int, default=8)
    p.add_argument("--min-remaining-ratio", type=float, default=0.50)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = ROOT / "runs/prune/experiment20_taylor_ladder_coco2017" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    info: dict[str, Any] = {"started_at": now(), "status": "starting",
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "cap_schedule": CAPS}
    log = (run_dir / "run.log").open("a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(old_out, log), Tee(old_err, log)
    try:
        weights = args.weights.resolve()
        if sha256(weights) != OFFICIAL_WEIGHT_SHA256:
            raise RuntimeError("Official YOLO11s hash mismatch")
        stage5_info = json.loads((args.stage5_run / "run_info.json").read_text(encoding="utf-8"))
        stage5 = Path(stage5_info["search_result"]["checkpoint"])
        if sha256(stage5) != stage5_info["search_result"]["checkpoint_sha256"]:
            raise RuntimeError("Stage-5 checkpoint hash mismatch")
        sensitivity_rows, sinfo = load_sensitivity(args.sensitivity_run, OFFICIAL_WEIGHT_SHA256)
        tiers = assign_tiers(sensitivity_rows, 0.40, 0.35)
        calibration = Path(sinfo["splits"]["calibration_yaml"])
        tune = Path(sinfo["splits"]["tune_yaml"])
        recovery = Path(sinfo["splits"]["recovery_yaml"])
        calibration8 = subset_yaml(recovery, run_dir / "bn_calibration_8192.yaml", 8192, args.seed)
        full = run_dir / "coco2017_runtime.yaml"
        make_runtime_data_yaml(args.data, args.dataset_root, full)
        parent = YOLO(str(weights)).model.float()
        initial = {name: module.out_channels for name, module in parent.named_modules() if isinstance(module, nn.Conv2d)}
        base_example = torch.zeros(1, 3, args.imgsz, args.imgsz)
        base_params, base_gmacs = stats(parent, base_example)
        del parent, base_example
        info.update(status="preflight_complete", source={"path": str(weights), "sha256": sha256(weights)},
                    stage5={"path": str(stage5), "sha256": sha256(stage5)}, tiers=tiers,
                    baseline={"parameters": base_params, "gmacs": base_gmacs})
        write_json(run_dir / "run_info.json", info)
        print("PREFLIGHT_COMPLETE", flush=True)
        if not args.execute:
            return 0

        device = torch.device("cuda:0")
        model = YOLO(str(stage5)).model.float().to(device).eval()
        identities = {name: list(range(module.out_channels)) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)}
        example = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
        _, current_gmacs = stats(model, example)
        accepted: list[dict[str, Any]] = []
        endpoints: list[dict[str, Any]] = []
        eval_args = SimpleNamespace(imgsz=args.imgsz, val_batch=args.val_batch, device=args.device, workers=args.workers, seed=args.seed)
        trials = run_dir / "candidates.csv"
        total_step = 0

        for target in (0.10, 0.15, 0.20):
            caps = stage_caps(initial, tiers, target, args.channel_step)
            while 1.0 - current_gmacs / base_gmacs < target:
                total_step += 1
                if total_step > args.max_steps:
                    raise RuntimeError("Maximum ladder steps reached")
                modules = dict(model.named_modules())
                active = [{"layer": name, "initial_channels": modules[name].out_channels}
                          for name in tiers if isinstance(modules.get(name), nn.Conv2d)
                          and initial[name] - modules[name].out_channels < caps[name]]
                if not active:
                    raise RuntimeError(f"No active layers before {target:.0%}")
                score_source = run_dir / "taylor_current.pt"
                save_model(model, score_source, stage5)
                scores, summary = collect_taylor_scores(score_source, calibration, active, device, args)
                write_json(run_dir / f"taylor_step_{total_step:03d}.json", summary)
                best = None
                for item in active:
                    name, tier = item["layer"], tiers[item["layer"]]
                    root = dict(model.named_modules())[name]
                    removed = initial[name] - root.out_channels
                    count = min(args.channel_step, caps[name] - removed)
                    row = {"stage": target, "step": total_step, "layer": name, "tier": tier, "layer_cap": caps[name]}
                    ranking = sorted(range(root.out_channels), key=lambda index: (scores[name][index], index))
                    indices = sorted(ranking[:count])
                    cost = sum(scores[name][index] for index in indices)
                    trial = copy.deepcopy(model)
                    try:
                        new_ids, change = prune_action(trial, name, indices, identities, example, initial, args.min_remaining_ratio)
                        verify_caps(trial, initial, caps)
                        _, gmacs = stats(trial, example)
                        saved = current_gmacs - gmacs
                        if saved <= 1e-9:
                            raise ValueError("no_positive_GMAC_saving")
                        row.update(status="feasible", reason="", before_channels=change["before_channels"],
                                   after_channels=change["after_channels"], removed_channels=change["removed_channels"],
                                   layer_removed_total=initial[name] - change["after_channels"], taylor_cost=cost,
                                   gmacs=gmacs, step_gmac_saved=saved,
                                   total_compute_reduction=1.0 - gmacs / base_gmacs, score_per_gmac=cost / saved)
                        candidate = {"row": row, "model": trial, "ids": new_ids}
                        if best is None or (row["score_per_gmac"], -saved, name) < (best["row"]["score_per_gmac"], -best["row"]["step_gmac_saved"], best["row"]["layer"]):
                            if best is not None:
                                del best["model"]
                            best = candidate
                        else:
                            del trial
                    except (ValueError, RuntimeError, KeyError, IndexError, AssertionError) as error:
                        row.update(status="rejected", reason=f"{type(error).__name__}: {error}")
                        del trial
                    append_row(trials, row)
                    gc.collect()
                score_source.unlink(missing_ok=True)
                if best is None:
                    raise RuntimeError(f"No feasible action before {target:.0%}")
                model, identities, current_gmacs = best["model"], best["ids"], best["row"]["gmacs"]
                chosen = dict(best["row"], status="accepted")
                accepted.append(chosen)
                info.update(status="pruning", accepted=accepted, endpoints=endpoints)
                save_model(model, run_dir / "pruned_partial.pt", stage5)
                write_json(run_dir / "run_info.json", info)
                print(f"ACCEPT stage={target:.0%} step={total_step} layer={chosen['layer']} reduction={chosen['total_compute_reduction']:.4%}", flush=True)
                torch.cuda.empty_cache()

            backward_check(model, device)
            result = endpoint(model, stage5, target, run_dir, calibration8, tune, full, args, eval_args)
            result["gmacs"] = current_gmacs
            result["compute_reduction"] = 1.0 - current_gmacs / base_gmacs
            endpoints.append(result)
            info["endpoints"] = endpoints
            write_json(run_dir / "run_info.json", info)
            print(f"ENDPOINT {target:.0%} reduction={result['compute_reduction']:.4%} full_map={result['selected']['full_val']['map50_95']:.6f}", flush=True)

        info.update(status="complete", finished_at=now(), accepted=accepted, endpoints=endpoints)
        write_json(run_dir / "run_info.json", info)
        lines = ["# Experiment 20: nested Taylor pruning ladder", "", "| Target | Actual GMAC reduction | Selected | COCO val mAP50-95 |", "|---:|---:|---|---:|"]
        for item in endpoints:
            lines.append(f"| {item['target']:.0%} | {item['compute_reduction']:.2%} | {item['selected']['label']} | {item['selected']['full_val']['map50_95']:.4f} |")
        (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"RUN_COMPLETE {run_dir}", flush=True)
        return 0
    except Exception as error:
        info.update(status="failed", finished_at=now(), error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        write_json(run_dir / "run_info.json", info)
        traceback.print_exc()
        return 1
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
