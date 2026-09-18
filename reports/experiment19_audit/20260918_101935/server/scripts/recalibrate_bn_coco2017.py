"""Compare weight-free BatchNorm recalibration recipes for a pruned YOLO model."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

from diagnose_coco2017_recovery import evaluate, make_runtime_data_yaml, sha256, write_json
from prune_greedy_coco2017 import Tee, save_model, shape_signature

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def recalibrate(source: Path, data: Path, output: Path, images: int, batch_size: int,
                workers: int, device: torch.device, seed: int, momentum: float | None,
                reset: bool) -> dict:
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 640, "rect": False})
    payload = check_det_dataset(str(data), autodownload=False)
    dataset = build_yolo_dataset(cfg, payload["train"], batch_size, payload, mode="val", rect=False)
    count = min(images, len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count].tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=batch_size,
        shuffle=False, num_workers=workers, collate_fn=dataset.collate_fn, pin_memory=True)
    yolo = YOLO(str(source))
    model = yolo.model.float().to(device)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    original_shape = shape_signature(model)
    model.eval()
    bn_count = 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_count += 1
            if reset:
                module.reset_running_stats()
            module.momentum = momentum
            module.train()
    seen = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            tensor = batch["img"].to(device, non_blocking=True).float() / 255.0
            model(tensor)
            seen += tensor.shape[0]
    if shape_signature(model) != original_shape:
        raise RuntimeError("BN recalibration changed model architecture")
    after = model.state_dict()
    illegal = []
    changed_bn = []
    for key, old in before.items():
        new = after[key].detach().cpu()
        if torch.equal(old, new):
            continue
        if key.endswith(("running_mean", "running_var", "num_batches_tracked")):
            changed_bn.append(key)
        else:
            illegal.append(key)
    if illegal or not changed_bn:
        raise RuntimeError(f"Unexpected state changes={illegal[:5]}, BN changes={len(changed_bn)}")
    model.eval()
    save_model(model, output, source)
    reloaded = YOLO(str(output)).model
    if shape_signature(reloaded) != original_shape:
        raise RuntimeError("Recalibrated checkpoint failed reload verification")
    return {"images": seen, "bn_layers": bn_count, "changed_bn_buffers": len(changed_bn),
            "momentum": momentum, "reset": reset, "seconds": time.perf_counter() - started,
            "sha256": sha256(output)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pruning-run", type=Path, required=True)
    p.add_argument("--sensitivity-run", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "configs/coco2017.yaml")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--val-batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extended", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def subset_yaml(source_yaml: Path, destination: Path, count: int, seed: int) -> Path:
    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    manifest = Path(payload["train"])
    lines = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    order = torch.randperm(len(lines), generator=torch.Generator().manual_seed(seed))[:count].tolist()
    subset = destination.with_suffix(".txt")
    subset.write_text("\n".join(lines[index] for index in order) + "\n", encoding="utf-8")
    payload["train"] = str(subset)
    destination.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destination


def main() -> int:
    args = parse_args()
    run_dir = ROOT / "runs/recovery/experiment19_bn_recalibration_coco2017" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    info = {"started_at": now(), "status": "starting", "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    log = (run_dir / "run.log").open("a", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(old_out, log), Tee(old_err, log)
    try:
        pinfo = json.loads((args.pruning_run / "run_info.json").read_text(encoding="utf-8"))
        sinfo = json.loads((args.sensitivity_run / "run_info.json").read_text(encoding="utf-8"))
        raw = Path(pinfo["search_result"]["checkpoint"])
        if sha256(raw) != pinfo["search_result"]["checkpoint_sha256"]:
            raise RuntimeError("Pruned checkpoint hash mismatch")
        tune = Path(sinfo["splits"]["tune_yaml"])
        calibration = Path(sinfo["splits"]["calibration_yaml"])
        recovery = Path(sinfo["splits"]["recovery_yaml"])
        full = run_dir / "coco2017_runtime.yaml"
        make_runtime_data_yaml(args.data, args.dataset_root, full)
        info.update(status="preflight_complete", source={"path": str(raw), "sha256": sha256(raw)})
        write_json(run_dir / "run_info.json", info)
        print("PREFLIGHT_COMPLETE", flush=True)
        if not args.execute:
            return 0
        eval_args = SimpleNamespace(imgsz=640, val_batch=args.val_batch, device=args.device, workers=args.workers, seed=args.seed)
        raw_metrics = evaluate(raw, tune, run_dir / "validation/raw", eval_args)
        if args.extended:
            data8 = subset_yaml(recovery, run_dir / "calibration_8192.yaml", 8192, args.seed)
            data16 = subset_yaml(recovery, run_dir / "calibration_16384.yaml", 16384, args.seed)
            data32 = subset_yaml(recovery, run_dir / "calibration_32768.yaml", 32768, args.seed)
            variants = [
                ("preserve_m00025_8192", data8, 8192, 0.0025, False),
                ("preserve_m0005_8192", data8, 8192, 0.005, False),
                ("preserve_m002_8192", data8, 8192, 0.02, False),
                ("preserve_m00025_16384", data16, 16384, 0.0025, False),
                ("preserve_m0005_16384", data16, 16384, 0.005, False),
                ("preserve_m001_16384", data16, 16384, 0.01, False),
                ("preserve_m00025_32768", data32, 32768, 0.0025, False),
                ("preserve_m0005_32768", data32, 32768, 0.005, False),
            ]
        else:
            variants = [
                ("preserve_m001_2048", calibration, 2048, 0.01, False),
                ("preserve_m01_2048", calibration, 2048, 0.1, False),
                ("reset_cma_2048", calibration, 2048, None, True),
                ("preserve_m001_8192", recovery, 8192, 0.01, False),
            ]
        results = [{"label": "raw", "path": str(raw), "sha256": sha256(raw), "tune": raw_metrics}]
        device = torch.device("cuda:0")
        for label, source_data, images, momentum, reset in variants:
            output = run_dir / f"{label}.pt"
            details = recalibrate(raw, source_data, output, images, args.batch, args.workers, device, args.seed, momentum, reset)
            metrics = evaluate(output, tune, run_dir / f"validation/{label}", eval_args)
            results.append({"label": label, "path": str(output), "sha256": details["sha256"], "tune": metrics, "details": details})
            write_json(run_dir / "partial_results.json", results)
        selected = max(results, key=lambda item: item["tune"]["map50_95"])
        selected_full = evaluate(Path(selected["path"]), full, run_dir / "validation_full/selected", eval_args)
        info.update(status="complete", finished_at=now(), results=results,
            selected={**selected, "full_val": selected_full})
        write_json(run_dir / "run_info.json", info)
        (run_dir / "report.md").write_text("# Experiment 19: BN recalibration\n\n" +
            "\n".join(f"- {r['label']}: tune mAP50-95 {r['tune']['map50_95']:.6f}" for r in results) +
            f"\n\nSelected: `{selected['label']}`, full COCO val mAP50-95 {selected_full['map50_95']:.6f}.\n", encoding="utf-8")
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
