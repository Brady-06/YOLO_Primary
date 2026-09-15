"""Recovery-only training for a tiered COCO2017 pruning checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from diagnose_coco2017_recovery import evaluate, make_runtime_data_yaml, sha256, train_one, write_json
from prune_greedy_coco2017 import Tee

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pruning-run", type=Path, required=True)
    parser.add_argument("--sensitivity-run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "configs/coco2017.yaml")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--val-batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = ROOT / "runs/recovery/experiment18_tiered_taylor_coco2017" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    info = {"started_at": now(), "status": "starting", "script": str(Path(__file__).resolve()), "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    log = (run_dir / "run.log").open("a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(old_out, log), Tee(old_err, log)
    started = time.perf_counter()
    try:
        pruning_run = args.pruning_run.resolve()
        sensitivity_run = args.sensitivity_run.resolve()
        pruning_info = json.loads((pruning_run / "run_info.json").read_text(encoding="utf-8"))
        sensitivity_info = json.loads((sensitivity_run / "run_info.json").read_text(encoding="utf-8"))
        if pruning_info.get("status") != "complete" or sensitivity_info.get("status") != "complete":
            raise RuntimeError("Input experiment is incomplete")
        raw = Path(pruning_info["search_result"]["checkpoint"])
        if sha256(raw) != pruning_info["search_result"]["checkpoint_sha256"]:
            raise RuntimeError("Pruned checkpoint hash mismatch")
        if pruning_info["source"]["sha256"] != sensitivity_info["source"]["sha256"]:
            raise RuntimeError("Source checkpoint lineage mismatch")
        recovery_yaml = Path(sensitivity_info["splits"]["recovery_yaml"])
        tune_yaml = Path(sensitivity_info["splits"]["tune_yaml"])
        full_yaml = run_dir / "coco2017_runtime.yaml"
        make_runtime_data_yaml(args.data, args.dataset_root, full_yaml)
        info.update(status="preflight_complete", lineage={"pruned": str(raw), "pruned_sha256": sha256(raw), "source_sha256": pruning_info["source"]["sha256"]}, splits={"recovery": str(recovery_yaml), "tune": str(tune_yaml), "full_val": str(full_yaml)})
        write_json(run_dir / "run_info.json", info)
        print("PREFLIGHT_COMPLETE", flush=True)
        if not args.execute:
            return 0

        common = SimpleNamespace(preset="bn_update", epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, val_batch=args.val_batch, workers=args.workers, device=args.device, seed=args.seed)
        info["status"] = "training"
        write_json(run_dir / "run_info.json", info)
        source_tune = evaluate(raw, tune_yaml, run_dir / "validation_tune/source", common)
        best, last, settings = train_one(raw, recovery_yaml, run_dir / "training/pruned", common)
        candidates = [{"label": "source", "path": raw, "metrics": source_tune}]
        for label, path in (("best", best), ("last", last)):
            candidates.append({"label": label, "path": path, "metrics": evaluate(path, tune_yaml, run_dir / f"validation_tune/{label}", common)})
        selected = max(candidates, key=lambda item: item["metrics"]["map50_95"])
        full_metrics = evaluate(selected["path"], full_yaml, run_dir / "validation_full/selected", common)
        info.update(
            status="complete", finished_at=now(), elapsed_seconds=time.perf_counter() - started,
            candidates=[{"label": item["label"], "path": str(item["path"]), "sha256": sha256(item["path"]), "tune": item["metrics"]} for item in candidates],
            selected={"label": selected["label"], "path": str(selected["path"]), "sha256": sha256(selected["path"]), "tune": selected["metrics"], "full_val": full_metrics},
            training_settings=settings,
        )
        write_json(run_dir / "run_info.json", info)
        (run_dir / "report.md").write_text(
            "# Experiment 18: conservative recovery\n\n"
            f"- Selected checkpoint: `{selected['label']}`\n"
            f"- Tune mAP50-95: {selected['metrics']['map50_95']:.4f}\n"
            f"- Full COCO val mAP50-95: {full_metrics['map50_95']:.4f}\n"
            f"- Recovery preset: `bn_update`, epochs: {args.epochs}\n",
            encoding="utf-8",
        )
        print(f"RUN_COMPLETE {run_dir}", flush=True)
        return 0
    except Exception as error:
        info.update(status="failed", finished_at=now(), elapsed_seconds=time.perf_counter() - started, error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        write_json(run_dir / "run_info.json", info)
        traceback.print_exc()
        return 1
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
