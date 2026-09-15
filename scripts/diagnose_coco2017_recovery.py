"""COCO2017 recovery diagnostic with strict checkpoint lineage checks.

This script is intentionally separate from experiments 07-10. It never overwrites
old artifacts. Without --execute it performs a CPU preflight only. A formal run
requires the exact unpruned parent checkpoint used to create the pruned checkpoint.
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
DEFAULT_PRUNED = (
    ROOT
    / "runs/prune/experiment10_coco2017_greedy/20260909_133307/pruned_raw.pt"
)
DEFAULT_PROVENANCE = (
    ROOT
    / "runs/prune/experiment10_coco2017_greedy/20260909_133307/run_info.json"
)

PRESETS: dict[str, dict[str, Any]] = {
    # Exact experiment-10 recovery settings. Kept only for reproducibility.
    "legacy_frozen": {
        "freeze_bn": True,
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 0.1,
        "warmup_epochs": 0.0,
        "warmup_bias_lr": 0.0001,
        "weight_decay": 0.0001,
        "mosaic": 0.0,
        "fliplr": 0.0,
        "scale": 0.0,
        "translate": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    },
    # Changes only the BN policy relative to experiment 10.
    "bn_update": {
        "freeze_bn": False,
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 0.1,
        "warmup_epochs": 0.0,
        "warmup_bias_lr": 0.0001,
        "weight_decay": 0.0001,
        "mosaic": 0.0,
        "fliplr": 0.0,
        "scale": 0.0,
        "translate": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    },
    # Secondary recovery candidate after the BN-only diagnostic.
    "standard_recovery": {
        "freeze_bn": False,
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
    },
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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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


def checkpoint(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"{label} must be an existing trusted .pt file: {path}")
    return path


def count_nonempty_lines(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def make_runtime_data_yaml(
    source: Path, dataset_root: Path, destination: Path
) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"COCO YAML does not exist: {source}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"COCO root does not exist: {dataset_root}")

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"COCO YAML must contain a mapping: {source}")
    for key in ("train", "val", "names"):
        if key not in payload:
            raise ValueError(f"COCO YAML is missing required key {key!r}")

    payload["path"] = dataset_root.as_posix()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    counts: dict[str, int | None] = {"train": None, "val": None}
    expected = {"train": 118287, "val": 5000}
    for split in ("train", "val"):
        split_value = payload[split]
        if isinstance(split_value, str) and split_value.endswith(".txt"):
            split_file = dataset_root / split_value
            if not split_file.is_file():
                raise FileNotFoundError(f"Missing COCO {split} list: {split_file}")
            counts[split] = count_nonempty_lines(split_file)
            if counts[split] != expected[split]:
                raise RuntimeError(
                    f"COCO {split} list has {counts[split]} entries; "
                    f"expected {expected[split]}"
                )
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "runtime": str(destination.resolve()),
        "runtime_sha256": sha256(destination),
        "dataset_root": str(dataset_root),
        "list_counts": counts,
    }


def model_shapes(model: nn.Module) -> dict[str, tuple[Any, ...]]:
    return {
        name: (
            type(module).__name__,
            tuple(
                (key, tuple(value.shape))
                for key, value in module.state_dict().items()
            ),
            getattr(module, "groups", None),
        )
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d))
    }


def bn_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{key}": value.detach().cpu().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
        for key, value in module.state_dict().items()
    }


def freeze_bn(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


class FrozenBNTrainer(DetectionTrainer):
    def _model_train(self) -> None:
        super()._model_train()
        freeze_bn(self.model)


def expected_parent_hash(provenance: Path) -> tuple[str, str]:
    provenance = Path(provenance).expanduser().resolve()
    if not provenance.is_file():
        raise FileNotFoundError(f"Provenance JSON does not exist: {provenance}")
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"Provenance JSON has no sources mapping: {provenance}")

    candidates = [
        (path, digest)
        for path, digest in sources.items()
        if str(path).replace("\\", "/").endswith(".pt")
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one parent .pt entry in provenance sources; "
            f"found {len(candidates)}"
        )
    path, digest = candidates[0]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Invalid parent SHA256 in provenance JSON")
    return str(path), digest.lower()


def verify_lineage(
    parent: Path, pruned: Path, provenance: Path
) -> dict[str, Any]:
    recorded_path, recorded_hash = expected_parent_hash(provenance)
    actual_parent_hash = sha256(parent)
    if actual_parent_hash != recorded_hash:
        raise RuntimeError(
            "Parent checkpoint SHA256 does not match pruning provenance. "
            f"Expected {recorded_hash}, got {actual_parent_hash}. "
            "Do not compare or recover unrelated checkpoints."
        )

    parent_model = YOLO(str(parent)).model
    pruned_model = YOLO(str(pruned)).model
    parent_params = sum(parameter.numel() for parameter in parent_model.parameters())
    pruned_params = sum(parameter.numel() for parameter in pruned_model.parameters())
    if pruned_params >= parent_params:
        raise RuntimeError(
            f"Expected pruned parameters < parent parameters, got "
            f"{pruned_params} >= {parent_params}"
        )
    if model_shapes(pruned_model) == model_shapes(parent_model):
        raise RuntimeError("Pruned and parent architectures unexpectedly match")

    result = {
        "recorded_parent_path": recorded_path,
        "expected_parent_sha256": recorded_hash,
        "actual_parent_sha256": actual_parent_hash,
        "pruned_sha256": sha256(pruned),
        "parent_parameters": parent_params,
        "pruned_parameters": pruned_params,
        "parameter_reduction": 1.0 - pruned_params / parent_params,
    }
    del parent_model, pruned_model
    return result


def evaluate(
    weights: Path,
    data: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
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
    numeric = [
        result["map50"],
        result["map50_95"],
        result["precision"],
        result["recall"],
        result["validation_inference_ms"],
    ]
    if not torch.isfinite(torch.tensor(numeric)).all().item():
        raise RuntimeError(f"Non-finite validation result for {weights}")
    del model, metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train_one(
    source: Path,
    data: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any]]:
    source_model = YOLO(str(source))
    expected_shapes = model_shapes(source_model.model)
    preset = dict(PRESETS[args.preset])
    freeze_policy = bool(preset.pop("freeze_bn"))
    trainer_type = FrozenBNTrainer if freeze_policy else DetectionTrainer

    overrides = {
        "model": str(source),
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.epochs,
        "close_mosaic": min(1, args.epochs),
        "amp": True,
        "plots": True,
        "seed": args.seed,
        "deterministic": True,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": False,
        **preset,
    }
    trainer = trainer_type(overrides=overrides)
    # Passing the module preserves structural pruning widths.
    trainer.model = source_model.model
    frozen_reference: dict[str, torch.Tensor] = {}

    def prepare(trainer_instance: DetectionTrainer) -> None:
        if freeze_policy:
            freeze_bn(trainer_instance.model)
            frozen_reference.update(bn_state(trainer_instance.model))
            if not frozen_reference:
                raise RuntimeError("Expected BatchNorm layers in recovery model")

    def check_batch(trainer_instance: DetectionTrainer) -> None:
        if (
            trainer_instance.loss is None
            or not torch.isfinite(trainer_instance.loss.detach()).all().item()
        ):
            raise RuntimeError("Recovery produced a missing or non-finite loss")
        if freeze_policy and any(
            module.training
            for module in trainer_instance.model.modules()
            if isinstance(module, nn.BatchNorm2d)
        ):
            raise RuntimeError("Frozen BatchNorm unexpectedly entered train mode")

    def check_final(trainer_instance: DetectionTrainer) -> None:
        if model_shapes(trainer_instance.model) != expected_shapes:
            raise RuntimeError("Recovery changed the model architecture")
        if freeze_policy:
            actual = bn_state(trainer_instance.model)
            changed = [
                key
                for key in frozen_reference
                if key not in actual
                or not torch.equal(frozen_reference[key], actual[key])
            ]
            if changed:
                raise RuntimeError(
                    f"Frozen BatchNorm state changed: {changed[:5]}"
                )

    trainer.callbacks["on_pretrain_routine_end"].append(prepare)
    trainer.callbacks["on_train_batch_end"].append(check_batch)
    trainer.callbacks["on_train_end"].append(check_final)
    trainer.train()

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
    return best, last, overrides


def write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "checkpoint",
        "selected",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "validation_inference_ms",
        "weights",
        "sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def report(run_dir: Path, info: dict[str, Any]) -> None:
    rows = info["results"]
    lines = [
        "# COCO2017 recovery diagnostic",
        "",
        f"- Preset: `{info['configuration']['preset']}`",
        f"- Epochs: `{info['configuration']['epochs']}`",
        "- Parent/pruned lineage was verified by SHA256 before training.",
        "- Selection includes the input checkpoint, so training cannot silently replace it with a worse best.pt.",
        "",
        "| Model | Input mAP50-95 | Selected mAP50-95 | Change | Selected checkpoint |",
        "|---|---:|---:|---:|---|",
    ]
    for label in ("parent", "pruned"):
        candidates = [row for row in rows if row["model"] == label]
        source = next(row for row in candidates if row["checkpoint"] == "source")
        selected = next(row for row in candidates if row["selected"])
        lines.append(
            f"| {label} | {source['map50_95']:.4f} | "
            f"{selected['map50_95']:.4f} | "
            f"{selected['map50_95'] - source['map50_95']:+.4f} | "
            f"{selected['checkpoint']} |"
        )
    lines.extend(
        [
            "",
            "This diagnostic decides whether recovery training is sound. "
            "It does not establish a pruning-method result.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace, run_dir: Path, info: dict[str, Any]) -> None:
    parent = checkpoint(args.parent, "Parent checkpoint")
    pruned = checkpoint(args.pruned, "Pruned checkpoint")
    provenance = Path(args.provenance).expanduser().resolve()
    data_info = make_runtime_data_yaml(
        args.data,
        args.dataset_root,
        run_dir / "coco2017_runtime.yaml",
    )
    lineage = verify_lineage(parent, pruned, provenance)
    info.update(
        {
            "data": data_info,
            "lineage": lineage,
            "configuration": {
                key: str(value) if isinstance(value, Path) else value
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
    print(json.dumps(lineage, indent=2), flush=True)

    if not args.execute:
        info["elapsed_seconds"] = time.perf_counter() - info.pop("_timer")
        write_json(run_dir / "run_info.json", info)
        print(
            "No training or validation was run. Pass --execute on the GPU server.",
            flush=True,
        )
        return
    if not torch.cuda.is_available() and str(args.device) != "cpu":
        raise RuntimeError("--execute requested a CUDA device, but CUDA is unavailable")

    runtime_data = Path(data_info["runtime"])
    all_rows: list[dict[str, Any]] = []
    training_settings: dict[str, Any] = {}
    for label, source in (("parent", parent), ("pruned", pruned)):
        candidates: list[dict[str, Any]] = []
        source_result = evaluate(
            source,
            runtime_data,
            run_dir / "validation" / label / "source",
            args,
        )
        source_result.update(
            {"model": label, "checkpoint": "source", "selected": False}
        )
        candidates.append(source_result)

        best, last, overrides = train_one(
            source,
            runtime_data,
            run_dir / "training" / label,
            args,
        )
        training_settings[label] = overrides
        for checkpoint_label, candidate_path in (("best", best), ("last", last)):
            result = evaluate(
                candidate_path,
                runtime_data,
                run_dir / "validation" / label / checkpoint_label,
                args,
            )
            result.update(
                {
                    "model": label,
                    "checkpoint": checkpoint_label,
                    "selected": False,
                }
            )
            candidates.append(result)

        selected = max(candidates, key=lambda row: row["map50_95"])
        selected["selected"] = True
        all_rows.extend(candidates)
        write_json(run_dir / "partial_results.json", all_rows)

    info.update(
        {
            "status": "complete",
            "finished_at": now(),
            "elapsed_seconds": time.perf_counter() - info.pop("_timer"),
            "results": all_rows,
            "training_settings": training_settings,
        }
    )
    write_comparison(run_dir / "comparison.csv", all_rows)
    report(run_dir, info)
    write_json(run_dir / "run_info.json", info)
    print(f"COMPLETE {run_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        type=Path,
        required=True,
        help="Exact unpruned checkpoint used to create --pruned.",
    )
    parser.add_argument("--pruned", type=Path, default=DEFAULT_PRUNED)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "configs/coco2017.yaml",
        help="Source COCO YAML; its path entry is replaced at runtime.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="COCO root containing train2017.txt and val2017.txt.",
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="bn_update")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--val-batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run full validation and training. Omit for CPU preflight.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.val_batch < 1:
        parser.error("epochs, batch, and val-batch must be positive")
    return args


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    run_dir = (
        ROOT
        / "runs/diagnostics/experiment14_coco2017_recovery"
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
            raise
        finally:
            for handler, stream in handlers:
                handler.setStream(stream)
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == "__main__":
    main()
