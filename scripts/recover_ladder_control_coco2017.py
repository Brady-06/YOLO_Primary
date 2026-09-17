"""实验19B control 训练：剪枝阶梯节点在 recovery 集上的普通微调恢复。

单个节点：
  source（BN 校准版） --20 epoch, standard_recovery, 有效 batch 128--> best.pt / last.pt
  在 tune 集（2048）上对 source/best/last 按 mAP50-95 选择
  选定 checkpoint 在 val2017（5000）上做最终评价

固定配方：standard_recovery（AdamW、lr0 1e-3、warmup 1、开增强、weight_decay 5e-4）。
只做普通检测训练（无教师），对应方案表中的 control。
本脚本不覆盖旧目录；每个节点/方案使用独立时间戳目录。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
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
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

ROOT = Path(__file__).resolve().parents[1]

# 每个节点的 source 必须是 BN 校准版，SHA256 与 19A 产物一致（防迁移损坏/用错权重）。
EXPECTED_SOURCE_SHA256 = {
    "5": "6cf7d188930892a5b13cd70760aca874f407c714dc92b14351e605f8cd2e21d2",
    "10": "bfb378d36810e63c0ee1b00e7b5bea931daddef7ce76affd2c1195aecafb6ced",
    "15": "6b6c29d091358a8cefbe6360315463bad1e25186c012a4ff8850e39c73e8e675",
    "20": "41a4cbcf2717856b8b02dbe86074b2a5e61dec969e3d6287c9a89e2fd4ee30bc",
}

DEFAULT_DATA = (
    ROOT
    / "runs/analysis/experiment16_taylor_sensitivity_coco2017/20260915_122417/recovery.yaml"
)
DEFAULT_VAL_DATA = ROOT / "configs/coco2017.yaml"

PRESET = {
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.1,
    "warmup_epochs": 1.0,
    "warmup_bias_lr": 0.01,
    "weight_decay": 0.0005,
    "mosaic": 0.5,
    "fliplr": 0.5,
    "scale": 0.3,
    "translate": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Tee:
    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, text):
        self.stream.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


def count_nonempty_lines(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def make_runtime_yaml(source: Path, dataset_root: Path, destination: Path) -> dict[str, Any]:
    """改写 yaml 的 path 指向 dataset_root，并核对 train/val 列表行数。"""
    source = Path(source).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"YAML does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "names" not in payload:
        raise ValueError(f"YAML must contain a mapping with names: {source}")

    payload["path"] = dataset_root.as_posix()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "runtime": str(destination.resolve()),
        "runtime_sha256": sha256(destination),
        "dataset_root": str(dataset_root),
    }


def model_shapes(model: nn.Module) -> dict[str, tuple[Any, ...]]:
    return {
        name: (
            type(module).__name__,
            tuple((key, tuple(value.shape)) for key, value in module.state_dict().items()),
            getattr(module, "groups", None),
        )
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d))
    }


def model_stats(weights: Path) -> dict[str, Any]:
    model = YOLO(str(weights)).model
    params = sum(parameter.numel() for parameter in model.parameters())
    gmacs: Any = None
    try:
        from thop import profile

        model = model.eval().to("cpu")
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 640, 640),), verbose=False)
        gmacs = float(macs) / 1e9
    except Exception as exc:  # GMAC 不是关键，失败不中断整轮
        gmacs = f"thop_failed: {exc}"
    del model
    return {"parameters": params, "gmacs": gmacs}


def evaluate(weights: Path, data: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
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
        "weights": str(weights),
        "sha256": sha256(weights),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "validation_inference_ms": float(metrics.speed["inference"]),
    }
    numeric = [result[k] for k in ("map50", "map50_95", "precision", "recall", "validation_inference_ms")]
    if not torch.isfinite(torch.tensor(numeric)).all().item():
        raise RuntimeError(f"Non-finite validation result for {weights}")
    del model, metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train_control(source: Path, data: Path, output_dir: Path, args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    source_model = YOLO(str(source))
    expected_shapes = model_shapes(source_model.model)

    overrides = {
        "model": str(source),
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "nbs": args.nbs,
        "device": args.device,
        "workers": args.workers,
        "patience": args.epochs,
        "close_mosaic": args.close_mosaic,
        "amp": True,
        "plots": True,
        "seed": args.seed,
        "deterministic": True,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": False,
        **PRESET,
    }
    trainer = DetectionTrainer(overrides=overrides)
    # 直接放入已加载的 pruned model，保留结构化剪枝后的通道宽度。
    trainer.model = source_model.model

    def check_final(trainer_instance: DetectionTrainer) -> None:
        if model_shapes(trainer_instance.model) != expected_shapes:
            raise RuntimeError("Recovery changed the model architecture")

    trainer.callbacks["on_train_end"].append(check_final)

    torch.cuda.reset_peak_memory_stats(int(args.device))
    trainer.train()
    peak_vram = torch.cuda.max_memory_allocated(int(args.device)) / (1024 ** 3)

    best = Path(trainer.best).resolve()
    last = Path(trainer.last).resolve()
    for label, path in (("best", best), ("last", last)):
        if not path.is_file():
            raise RuntimeError(f"Recovery did not produce {label}.pt")
        reloaded = YOLO(str(path))
        if model_shapes(reloaded.model) != expected_shapes:
            raise RuntimeError(f"Reloaded {label}.pt changed pruned architecture")
        del reloaded

    del trainer, source_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best, last, overrides, peak_vram


def write_report(run_dir: Path, info: dict[str, Any]) -> None:
    cfg = info["configuration"]
    selected = next(row for row in info["results"] if row["checkpoint"] == info["selected_checkpoint"])
    source_row = next(row for row in info["results"] if row["checkpoint"] == "source")
    lines = [
        "# 实验19B control 恢复",
        "",
        f"- 节点：{cfg['node']}%（source tune mAP50-95 = {source_row['tune_map50_95']:.4f}）",
        f"- 配方：standard_recovery，{cfg['epochs']} epoch，有效 batch {cfg['nbs']}（物理 {cfg['batch']}）",
        f"- 选择：source/best/last 在 tune 集（2048）按 mAP50-95，选定 `{info['selected_checkpoint']}`",
        f"- 选定 checkpoint 在 val2017 的 mAP50-95 = {selected['val_map50_95']:.4f}",
        "",
        "| checkpoint | tune mAP50-95 | val2017 mAP50-95 | selected |",
        "|---|---:|---:|---|",
    ]
    for row in info["results"]:
        lines.append(
            f"| {row['checkpoint']} | {row['tune_map50_95']:.4f} | {row['val_map50_95']:.4f} | "
            f"{'yes' if row['selected'] else ''} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace, run_dir: Path, info: dict[str, Any]) -> None:
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")

    actual = sha256(source)
    expected = EXPECTED_SOURCE_SHA256.get(args.node)
    if expected is None:
        raise ValueError(f"Unknown node label {args.node!r}; expected one of {sorted(EXPECTED_SOURCE_SHA256)}")
    if actual != expected:
        raise RuntimeError(
            f"Source SHA256 mismatch for node {args.node}: expected {expected}, got {actual}. "
            "Do not train from an unverified checkpoint."
        )

    train_data_info = make_runtime_yaml(args.data, args.dataset_root, run_dir / "recovery_runtime.yaml")
    val_data_info = make_runtime_yaml(args.val_data, args.dataset_root, run_dir / "val_runtime.yaml")

    info.update(
        {
            "source": {"path": str(source), "sha256": actual},
            "source_stats": model_stats(source),
            "data": {"train": train_data_info, "val": val_data_info},
            "configuration": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": importlib.metadata.version("torch"),
                "ultralytics": importlib.metadata.version("ultralytics"),
                "cuda_available": torch.cuda.is_available(),
                "gpu": (
                    torch.cuda.get_device_name(int(args.device))
                    if torch.cuda.is_available() and str(args.device).isdigit()
                    else None
                ),
            },
            "status": "preflight_complete",
        }
    )
    write_json(run_dir / "run_info.json", info)
    print("PREFLIGHT_COMPLETE", flush=True)

    if not args.execute:
        info["elapsed_seconds"] = time.perf_counter() - info.pop("_timer")
        write_json(run_dir / "run_info.json", info)
        print("No training/validation run. Pass --execute on the GPU server.", flush=True)
        return
    if not torch.cuda.is_available() and str(args.device) != "cpu":
        raise RuntimeError("--execute requested a CUDA device, but CUDA is unavailable")

    # 1) source 在 tune 集上的基线（选择候选之一）
    source_tune = evaluate(source, Path(train_data_info["runtime"]), run_dir / "validation" / "source_tune", args)

    # 2) 训练
    best, last, overrides, peak_vram = train_control(
        source, Path(train_data_info["runtime"]), run_dir / "training", args
    )
    info["training_settings"] = overrides
    info["peak_vram_gb"] = peak_vram

    # 3) best/last 在 tune 集上评价
    best_tune = evaluate(best, Path(train_data_info["runtime"]), run_dir / "validation" / "best_tune", args)
    last_tune = evaluate(last, Path(train_data_info["runtime"]), run_dir / "validation" / "last_tune", args)

    candidates = [
        {"checkpoint": "source", "tune": source_tune, "weights": source},
        {"checkpoint": "best", "tune": best_tune, "weights": best},
        {"checkpoint": "last", "tune": last_tune, "weights": last},
    ]
    winner = max(candidates, key=lambda row: row["tune"]["map50_95"])

    # 4) 选定 checkpoint 在 val2017 上最终评价
    val_final = evaluate(
        winner["weights"], Path(val_data_info["runtime"]), run_dir / "validation" / "val2017_selected", args
    )

    # 汇总 comparison 行
    rows = []
    for candidate in candidates:
        row = {
            "node": args.node,
            "checkpoint": candidate["checkpoint"],
            "tune_map50_95": candidate["tune"]["map50_95"],
            "tune_map50": candidate["tune"]["map50"],
            "weights": str(candidate["weights"]),
            "sha256": candidate["tune"]["sha256"],
            "selected": candidate["checkpoint"] == winner["checkpoint"],
        }
        if candidate["checkpoint"] == winner["checkpoint"]:
            row["val_map50_95"] = val_final["map50_95"]
            row["val_map50"] = val_final["map50"]
            row["val_precision"] = val_final["precision"]
            row["val_recall"] = val_final["recall"]
        rows.append(row)

    info.update(
        {
            "status": "complete",
            "finished_at": now(),
            "elapsed_seconds": time.perf_counter() - info.pop("_timer"),
            "selected_checkpoint": winner["checkpoint"],
            "selected_stats": model_stats(winner["weights"]),
            "results": rows,
        }
    )

    fields = ["node", "checkpoint", "tune_map50_95", "tune_map50", "val_map50_95",
              "val_map50", "val_precision", "val_recall", "weights", "sha256", "selected"]
    with (run_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({f: row.get(f, "") for f in fields} for row in rows)

    write_report(run_dir, info)
    write_json(run_dir / "run_info.json", info)
    print(f"COMPLETE {run_dir} selected={winner['checkpoint']} val_map50_95={val_final['map50_95']:.4f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True, choices=sorted(EXPECTED_SOURCE_SHA256),
                        help="Pruning ladder node label (5/10/15/20).")
    parser.add_argument("--source", type=Path, required=True, help="Node source BN checkpoint (.pt).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Recovery YAML (train=recovery 114191, val=tune 2048).")
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA,
                        help="Full COCO2017 YAML (val=val2017 5000) for final evaluation.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/datasets/coco"),
                        help="COCO dataset root.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--nbs", type=int, default=128, help="Nominal (effective) batch size.")
    parser.add_argument("--val-batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true",
                        help="Run training and validation. Omit for CPU preflight.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.val_batch < 1 or args.nbs < 1:
        parser.error("epochs, batch, val-batch, nbs must be positive")
    return args


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    run_root = ROOT / "runs/recovery/experiment19b_coco2017" / f"node{args.node}_control"
    run_dir = run_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    info: dict[str, Any] = {
        "started_at": now(),
        "status": "starting",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
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
            run(args, run_dir, info)
        except BaseException as error:
            info.update(
                {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "finished_at": now(),
                    "elapsed_seconds": time.perf_counter() - info.pop("_timer", None),
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )
            write_json(run_dir / "run_info.json", info)
            raise
        finally:
            for handler, stream in handlers:
                handler.setStream(stream)
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == "__main__":
    main()
