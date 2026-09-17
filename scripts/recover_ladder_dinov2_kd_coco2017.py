"""实验19C DINOv2 教师蒸馏：剪枝阶梯入选节点 + 冻结 DINOv2 ViT-S/14 多尺度蒸馏。

学生：入选节点的 source（BN 校准版，结构化剪枝后的 YOLO11s）。
教师：冻结 DINOv2 ViT-S/14（torch.hub 官方入口），保持 eval()。
蒸馏：一次前向取第 4/8/12 个 Transformer Block（代码索引 3/7/11，reshape+norm），
      空间对齐 YOLO P3/P4/P5，1×1 投影到 384 通道，cosine distance。
复用 `scripts/distill_dinov2_coco128.py` 的多尺度蒸馏实现（不重写、不凭记忆编造接口）。

配方与 control / YOLO KD 完全一致（standard_recovery、20 epoch、有效 batch 128），
唯一差别是额外 DINOv2 蒸馏损失。部署 checkpoint 只保留普通 YOLO 学生。
"""

from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as F
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SOURCE_SHA256 = {
    "5": "6cf7d188930892a5b13cd70760aca874f407c714dc92b14351e605f8cd2e21d2",
    "10": "bfb378d36810e63c0ee1b00e7b5bea931daddef7ce76affd2c1195aecafb6ced",
    "15": "6b6c29d091358a8cefbe6360315463bad1e25186c012a4ff8850e39c73e8e675",
    "20": "41a4cbcf2717856b8b02dbe86074b2a5e61dec969e3d6287c9a89e2fd4ee30bc",
}
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
DINO_WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DINO_HUB_WEIGHTS_PATH = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"

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


def make_runtime_yaml(source: Path, dataset_root: Path, destination: Path) -> dict[str, Any]:
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
        and not name.startswith("_dino_feature_distiller")
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
    except Exception as exc:
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


# ---------------------------------------------------------------------------
# DINOv2 多尺度蒸馏机制（逐行复用 distill_dinov2_coco128.py 的实现）
# ---------------------------------------------------------------------------

class CaptureFeature:
    """Pickle-safe hook：只存引用，序列化时清空捕获张量。"""

    def __init__(self, features: dict[str, torch.Tensor], key: str):
        self.features, self.key = features, key

    def __call__(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.features[self.key] = output

    def __getstate__(self):
        self.features.clear()
        return {"features": self.features, "key": self.key}


class MultiScaleDINOFeatureDistiller(nn.Module):
    """一次前向取 DINOv2 块 4/8/12，对齐 YOLO P3/P4/P5。"""

    def __init__(self, student_channels: list[int], device: torch.device, dino_size: int, weight: float):
        super().__init__()
        if dino_size % 14:
            raise ValueError("--dino-size must be divisible by DINOv2 patch size 14")
        if len(student_channels) != 3:
            raise ValueError("P3/P4/P5 distillation requires exactly three student features")
        self.dino_size = dino_size
        self.base_weight = weight
        self.current_weight = 0.0
        self.block_indices = (3, 7, 11)  # zero-based：transformer blocks 4/8/12
        self.level_weights = (0.25, 0.50, 0.25)
        self.teacher = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.projectors = nn.ModuleList(
            nn.Conv2d(channels, 384, kernel_size=1, bias=False) for channels in student_channels
        ).to(device)
        for projector in self.projectors:
            nn.init.kaiming_normal_(projector.weight, mode="fan_out", nonlinearity="linear")
        self.register_buffer("mean", torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def teacher_features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        images = F.interpolate(images, size=(self.dino_size, self.dino_size), mode="bilinear", align_corners=False)
        normalized = (images - self.mean) / self.std
        return self.teacher.get_intermediate_layers(
            normalized, n=self.block_indices, reshape=True, norm=True
        )

    def loss(self, student_features: list[torch.Tensor], images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        teacher_features = self.teacher_features(images)
        level_losses = []
        for student, teacher, projector in zip(student_features, teacher_features, self.projectors):
            teacher = F.interpolate(teacher, size=student.shape[-2:], mode="bilinear", align_corners=False)
            projected = F.normalize(projector(student), dim=1)
            teacher = F.normalize(teacher, dim=1)
            level_losses.append(1.0 - (projected * teacher).sum(dim=1).mean())
        total = sum(level_weight * loss for level_weight, loss in zip(self.level_weights, level_losses))
        return total, level_losses


class MultiScaleDINOStudentModel(DetectionModel):
    """DetectionModel 附加 DINOv2 P3/P4/P5 特征损失。"""

    def loss(self, batch: dict, preds=None):
        regular_loss, loss_items = super().loss(batch, preds)
        if not self.training:
            return regular_loss, loss_items
        features = self._dino_student_features
        if any(level not in features for level in ("p3", "p4", "p5")):
            raise RuntimeError("P3/P4/P5 hooks did not capture all student features")
        dino_loss, level_losses = self._dino_feature_distiller.loss(
            [features["p3"], features["p4"], features["p5"]], batch["img"]
        )
        loss_items["dino_loss"] = dino_loss.detach()
        for level, level_loss in zip(("p3", "p4", "p5"), level_losses):
            loss_items[f"dino_{level}"] = level_loss.detach()
        scaled = dino_loss * self._dino_feature_distiller.current_weight * batch["img"].shape[0]
        return torch.cat((regular_loss, scaled.reshape(1))), loss_items


# 让训练期 checkpoint 可被 pickle 按名字反查类（脚本以 `python scripts/xxx.py` 运行时补注册父包）。
import types as _types
_SERIALIZATION_MODULE = "scripts.recover_ladder_dinov2_kd_coco2017"
if "scripts" not in sys.modules:
    _parent_module = _types.ModuleType("scripts")
    _parent_module.__path__ = []
    sys.modules["scripts"] = _parent_module
sys.modules.setdefault(_SERIALIZATION_MODULE, sys.modules[__name__])
for _serializable_class in (CaptureFeature, MultiScaleDINOFeatureDistiller, MultiScaleDINOStudentModel):
    _serializable_class.__module__ = _SERIALIZATION_MODULE


def _find_pyramid_layers(model: nn.Module) -> list[nn.Module]:
    detect = model.model[-1]
    sources = getattr(detect, "f", None)
    if not isinstance(sources, (list, tuple)) or len(sources) != 3:
        raise RuntimeError("Could not identify Detect's P3/P4/P5 input layers")
    return [model.model[index] for index in sources]


def attach_multiscale_distillation(model: nn.Module, device: torch.device, dino_size: int, weight: float) -> None:
    captured: dict[str, torch.Tensor] = {}
    levels = ("p3", "p4", "p5")
    pyramid_layers = _find_pyramid_layers(model)
    probed: dict[str, torch.Tensor] = {}
    handles = []
    with torch.no_grad():
        model.eval()
        for level, layer in zip(levels, pyramid_layers):
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output, key=level: probed.__setitem__(key, output)
                )
            )
        model(torch.zeros(1, 3, 640, 640, device=device))
        for handle in handles:
            handle.remove()
        model.train()
    if any(level not in probed or not isinstance(probed[level], torch.Tensor) for level in levels):
        raise RuntimeError("P3/P4/P5 feature probe failed")
    channels = [probed[level].shape[1] for level in levels]
    model.add_module("_dino_feature_distiller", MultiScaleDINOFeatureDistiller(channels, device, dino_size, weight))
    model._dino_student_features = captured
    for level, layer in zip(levels, pyramid_layers):
        layer.register_forward_hook(CaptureFeature(captured, level))
    model.__class__ = MultiScaleDINOStudentModel


def detach_for_inference(model: nn.Module) -> nn.Module:
    clean = copy.deepcopy(model).float().eval()
    for module in clean.modules():
        for hook_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, CaptureFeature):
                del module._forward_hooks[hook_id]
    clean.__dict__.pop("_dino_student_features", None)
    if "_dino_feature_distiller" in clean._modules:
        del clean._modules["_dino_feature_distiller"]
    clean.__class__ = DetectionModel
    return clean


def write_clean_checkpoint(wrapper_path: Path, clean_path: Path) -> None:
    checkpoint = torch.load(wrapper_path, map_location="cpu", weights_only=False)
    wrapped = checkpoint.get("ema") or checkpoint.get("model")
    checkpoint["ema"] = detach_for_inference(wrapped).half()
    checkpoint["model"] = None
    torch.save(checkpoint, clean_path)


class KDTrainer(DetectionTrainer):
    """标准训练 + 每次 build_optimizer / 进入 train 时重新冻结 DINOv2 教师。"""

    def _freeze_teacher(self) -> None:
        distiller = getattr(self.model, "_dino_feature_distiller", None)
        if distiller is not None:
            distiller.teacher.eval()
            for parameter in distiller.teacher.parameters():
                parameter.requires_grad_(False)

    def build_optimizer(self, *args, **kwargs):
        self._freeze_teacher()
        return super().build_optimizer(*args, **kwargs)

    def _model_train(self):
        super()._model_train()
        self._freeze_teacher()


def train_kd(
    source: Path, data: Path, output_dir: Path, args: argparse.Namespace
) -> tuple[Path, Path, dict[str, Any], float, str]:
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
    trainer = KDTrainer(overrides=overrides)
    trainer.model = source_model.model.to(trainer.device)
    attach_multiscale_distillation(trainer.model, trainer.device, args.dino_size, args.kd_weight)

    def prepare(trainer_instance: DetectionTrainer) -> None:
        trainer_instance.model._dino_feature_distiller.current_weight = args.kd_weight / max(1, args.kd_warmup)

    def set_weight(trainer_instance: DetectionTrainer) -> None:
        fraction = min(1.0, (trainer_instance.epoch + 1) / max(1, args.kd_warmup))
        trainer_instance.model._dino_feature_distiller.current_weight = args.kd_weight * fraction

    def check_final(trainer_instance: DetectionTrainer) -> None:
        if model_shapes(trainer_instance.model) != expected_shapes:
            raise RuntimeError("DINOv2 KD changed the student architecture")

    trainer.callbacks["on_pretrain_routine_end"].append(prepare)
    trainer.callbacks["on_train_epoch_start"].append(set_weight)
    trainer.callbacks["on_train_end"].append(check_final)

    torch.cuda.reset_peak_memory_stats(int(args.device))
    trainer.train()
    peak_vram = torch.cuda.max_memory_allocated(int(args.device)) / (1024 ** 3)

    wrapper_best = Path(trainer.best).resolve()
    wrapper_last = Path(trainer.last).resolve()
    for label, path in (("best", wrapper_best), ("last", wrapper_last)):
        if not path.is_file():
            raise RuntimeError(f"DINOv2 KD did not produce {label}.pt")
    clean_best = output_dir / "weights" / "best_clean.pt"
    clean_last = output_dir / "weights" / "last_clean.pt"
    clean_best.parent.mkdir(parents=True, exist_ok=True)
    write_clean_checkpoint(wrapper_best, clean_best)
    write_clean_checkpoint(wrapper_last, clean_last)

    teacher_weights_sha256 = sha256(DINO_HUB_WEIGHTS_PATH) if DINO_HUB_WEIGHTS_PATH.is_file() else "not_found"
    del trainer, source_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return clean_best, clean_last, overrides, peak_vram, teacher_weights_sha256


def write_report(run_dir: Path, info: dict[str, Any]) -> None:
    cfg = info["configuration"]
    selected = next(row for row in info["results"] if row["checkpoint"] == info["selected_checkpoint"])
    source_row = next(row for row in info["results"] if row["checkpoint"] == "source")
    lines = [
        "# 实验19C DINOv2 KD（入选节点）",
        "",
        f"- 节点：{cfg['node']}%（source tune mAP50-95 = {source_row['tune_map50_95']:.4f}）",
        f"- 教师：DINOv2 ViT-S/14（冻结，块 4/8/12→P3/P4/P5，1×1 投影 384 通道，cosine distance，权重 {cfg['kd_weight']}）",
        f"- 配方：standard_recovery，{cfg['epochs']} epoch，有效 batch {cfg['nbs']}（物理 {cfg['batch']}）",
        f"- 选择：source/best/last 在 tune 集按 mAP50-95，选定 `{info['selected_checkpoint']}`",
        "",
        "| checkpoint | tune mAP50-95 | val2017 mAP50-95 | selected |",
        "|---|---:|---:|---|",
    ]
    for row in info["results"]:
        val = f"{row['val_map50_95']:.4f}" if "val_map50_95" in row else ""
        lines.append(f"| {row['checkpoint']} | {row['tune_map50_95']:.4f} | {val} | {'yes' if row['selected'] else ''} |")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace, run_dir: Path, info: dict[str, Any]) -> None:
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")

    actual = sha256(source)
    expected = EXPECTED_SOURCE_SHA256.get(args.node)
    if expected is None:
        raise ValueError(f"Unknown node label {args.node!r}")
    if actual != expected:
        raise RuntimeError(f"Source SHA256 mismatch for node {args.node}: expected {expected}, got {actual}.")

    train_data_info = make_runtime_yaml(args.data, args.dataset_root, run_dir / "recovery_runtime.yaml")
    val_data_info = make_runtime_yaml(args.val_data, args.dataset_root, run_dir / "val_runtime.yaml")

    info.update(
        {
            "source": {"path": str(source), "sha256": actual},
            "teacher": {
                "model": "dinov2_vits14",
                "hub": "facebookresearch/dinov2",
                "expected_weights_sha256": DINO_WEIGHTS_SHA256,
                "feature_layers": "blocks 4/8/12 (zero-based 3/7/11), get_intermediate_layers(n=..., reshape=True, norm=True)",
                "input": "bilinear to dino_size, normalize DINO_MEAN/STD",
                "projection": "1x1 conv -> 384 channels, kaiming fan_out",
            },
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

    source_tune = evaluate(source, Path(train_data_info["runtime"]), run_dir / "validation" / "source_tune", args)

    best, last, overrides, peak_vram, teacher_weights_sha256 = train_kd(
        source, Path(train_data_info["runtime"]), run_dir / "training", args
    )
    info["training_settings"] = overrides
    info["peak_vram_gb"] = peak_vram
    info["teacher"]["actual_weights_sha256"] = teacher_weights_sha256
    if teacher_weights_sha256 != DINO_WEIGHTS_SHA256 and teacher_weights_sha256 != "not_found":
        LOGGER.warning(f"DINOv2 weights SHA256 mismatch: expected {DINO_WEIGHTS_SHA256}, got {teacher_weights_sha256}")

    best_tune = evaluate(best, Path(train_data_info["runtime"]), run_dir / "validation" / "best_tune", args)
    last_tune = evaluate(last, Path(train_data_info["runtime"]), run_dir / "validation" / "last_tune", args)

    candidates = [
        {"checkpoint": "source", "tune": source_tune, "weights": source},
        {"checkpoint": "best", "tune": best_tune, "weights": best},
        {"checkpoint": "last", "tune": last_tune, "weights": last},
    ]
    winner = max(candidates, key=lambda row: row["tune"]["map50_95"])

    val_final = evaluate(
        winner["weights"], Path(val_data_info["runtime"]), run_dir / "validation" / "val2017_selected", args
    )

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
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/datasets/coco"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--nbs", type=int, default=128)
    parser.add_argument("--val-batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--kd-weight", type=float, default=0.5, help="Final DINO loss weight.")
    parser.add_argument("--kd-warmup", type=int, default=2, help="DINO loss linear warmup epochs.")
    parser.add_argument("--dino-size", type=int, default=644, help="DINOv2 input size (multiple of 14).")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.val_batch < 1 or args.nbs < 1:
        parser.error("epochs, batch, val-batch, nbs must be positive")
    return args


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    run_root = ROOT / "runs/recovery/experiment19c_coco2017" / f"node{args.node}_dinov2_kd"
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
        handlers = [(handler, handler.stream) for handler in LOGGER.handlers if hasattr(handler, "stream")]
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
