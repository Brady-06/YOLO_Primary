"""实验19B control 结果收尾（无需重跑训练）。

背景：control 脚本早前版本 write_report 有 KeyError bug（selected['map50_95']），
导致训练 + 评估 + comparison.csv 全部完成后，在写 report.md / 最终 run_info.json 时崩溃。
本脚本从已写入的 comparison.csv 重建 report.md，并把 run_info.json 补成 complete 状态。

用法（在服务器 /root/YOLO 下）：
  /usr/local/miniconda3/envs/py312/bin/python scripts/finalize_control_report.py \
      --run-dir runs/recovery/experiment19b_coco2017/node10_control/20260917_055208
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


def now() -> str:
    return datetime.now().astimezone().isoformat()


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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"parameters": params, "gmacs": gmacs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()

    comparison = run_dir / "comparison.csv"
    if not comparison.is_file():
        raise FileNotFoundError(f"comparison.csv not found: {comparison}")

    with comparison.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected_rows = [row for row in rows if row.get("selected", "").strip() == "True"]
    if len(selected_rows) != 1:
        raise RuntimeError(f"Expected exactly one selected row, got {len(selected_rows)}")
    selected = selected_rows[0]
    selected_checkpoint = selected["checkpoint"]

    source_row = next(row for row in rows if row["checkpoint"] == "source")

    info_path = run_dir / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    cfg = info.get("configuration", {})

    # 把 CSV 的字符串数值还原成 float 存进 run_info（与正常完成路径一致）。
    normalized_rows = []
    for row in rows:
        normalized = {
            "node": row["node"],
            "checkpoint": row["checkpoint"],
            "tune_map50_95": float(row["tune_map50_95"]),
            "tune_map50": float(row["tune_map50"]),
            "weights": row["weights"],
            "sha256": row["sha256"],
            "selected": row.get("selected", "").strip() == "True",
        }
        for key in ("val_map50_95", "val_map50", "val_precision", "val_recall"):
            value = row.get(key, "").strip()
            if value:
                normalized[key] = float(value)
        normalized_rows.append(normalized)

    selected_norm = next(row for row in normalized_rows if row["selected"])
    selected_weights = Path(selected_norm["weights"])

    lines = [
        "# 实验19B control 恢复",
        "",
        f"- 节点：{cfg.get('node', selected_norm['node'])}%（source tune mAP50-95 = {float(source_row['tune_map50_95']):.4f}）",
        f"- 配方：standard_recovery，{cfg.get('epochs')} epoch，有效 batch {cfg.get('nbs')}（物理 {cfg.get('batch')}）",
        f"- 选择：source/best/last 在 tune 集（2048）按 mAP50-95，选定 `{selected_checkpoint}`",
        f"- 选定 checkpoint 在 val2017 的 mAP50-95 = {selected_norm['val_map50_95']:.4f}",
        "",
        "| checkpoint | tune mAP50-95 | val2017 mAP50-95 | selected |",
        "|---|---:|---:|---|",
    ]
    for row in normalized_rows:
        val = f"{row['val_map50_95']:.4f}" if "val_map50_95" in row else ""
        lines.append(
            f"| {row['checkpoint']} | {row['tune_map50_95']:.4f} | {val} | "
            f"{'yes' if row['selected'] else ''} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    info.update(
        {
            "status": "complete",
            "finished_at": now(),
            "selected_checkpoint": selected_checkpoint,
            "selected_stats": model_stats(selected_weights),
            "results": normalized_rows,
            "finalized_posthoc": True,
            "finalized_note": (
                "原脚本 write_report 在 comparison.csv 之后崩溃（KeyError: map50_95），"
                "训练与评估数据完整无损；本脚本从 comparison.csv 重建 report.md 并补全本状态。"
            ),
        }
    )
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FINALIZED {run_dir} selected={selected_checkpoint} "
          f"val_map50_95={selected_norm['val_map50_95']:.4f}")


if __name__ == "__main__":
    main()
