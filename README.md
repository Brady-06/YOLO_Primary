# YOLO11 结构化剪枝与 DINOv2 蒸馏

本仓库记录 YOLO11 在 COCO8、COCO128、COCO2017 上的训练、结构化剪枝，以及**对剪枝后模型进行 DINOv2 特征蒸馏**的实验。

## 核心研究顺序

```text
官方 YOLO 权重
      ↓
训练并建立未剪枝基线
      ↓
敏感度分析与结构化剪枝
      ↓
剪枝后微调，选出实验 05 greedy best.pt
      ↓
以该剪枝模型为学生进行 DINOv2 蒸馏
      ├── 实验 12：P4 单尺度
      └── 实验 13：P3/P4/P5 多尺度
      ↓
验证并导出不含教师的部署权重
```

因此，DINOv2 **不参与剪枝过程**；它只在剪枝结束后作为冻结教师，尝试帮助剪枝模型恢复或保持精度。

当前阶段已经完成：

- COCO128 基线训练与固定划分；
- L1 masking 逐层敏感度分析；
- 独立法、贪心法、梯度分档结构化剪枝；
- YOLO11s / COCO2017 基线、敏感度和贪心剪枝尝试；
- 剪枝 YOLO11n 的 P4 单尺度 DINOv2 蒸馏；
- P3/P4/P5 多尺度 DINOv2 蒸馏；
- 实验报告、运行记录与学习笔记整理。

当前推荐继续研究的学生模型是实验 05 贪心剪枝 `best.pt`。蒸馏尚未刷新其峰值精度，但能缓解普通微调的后期退化。

## 快速导航

| 内容 | 位置 |
| --- | --- |
| 学习资料总览 | [Summary & Learning/README.md](<Summary & Learning/README.md>) |
| 项目结构与数据流 | [项目结构与数据流讲解.ipynb](<Summary & Learning/项目结构与数据流讲解.ipynb>) |
| 剪枝原理与代码 | [pruning.ipynb](<Summary & Learning/pruning.ipynb>) |
| DINOv2 蒸馏原理与代码 | [distillation.ipynb](<Summary & Learning/distillation.ipynb>) |
| 实验报告 | [reports/](reports/) |
| 实验脚本 | [scripts/](scripts/) |
| 数据配置 | [configs/](configs/) |

建议学习顺序：项目结构与数据流 → 剪枝 → 蒸馏 → 对照正式报告和脚本。

## 实验路线

| 阶段 | 数据 + 模型 | 实验 | 内容 |
| --- | --- | --- | --- |
| 1 | COCO8 + YOLO11n | 01 | 环境与训练/验证流程验证 |
| 2 | COCO128 + YOLO11n | 02–06 | 基线、敏感度、独立剪枝、贪心剪枝、梯度分档 |
| 3 | COCO2017 + YOLO11s | 07–10 | 官方基线、训练尝试、敏感度、贪心剪枝 |
| 4 | COCO128 + 剪枝后的 YOLO11n + DINOv2 | 12–13 | 对实验 05 剪枝模型进行 P4 单尺度和 P3/P4/P5 多尺度特征蒸馏 |
| 5 | COCO2017 + YOLO11s | 14–15 | 模型谱系核对、恢复诊断与正式Taylor剪枝入口 |

实验 11 是结构化剪枝方法与代码速查，并包含 GroupNorm 等探索，不作为统一口径的正式性能结论。

实验 14 尚未运行训练。计划、基线哈希和服务器运行边界见
[`experiment14_coco2017_recovery_plan.md`](reports/experiment14_coco2017_recovery_plan.md)，
恢复诊断入口为 `scripts/diagnose_coco2017_recovery.py`。
正式Taylor入口为 `scripts/prune_taylor_coco2017.py`；两者默认只做预检，
只有显式传入 `--execute` 才会启动完整COCO验证、训练或剪枝。

## 环境

本机实验环境：

- Windows 11
- Python 3.13
- PyTorch 2.11 + CUDA 12.8
- Ultralytics 8.4
- NVIDIA GeForce RTX 5060 Ti 16 GB
- Torch-Pruning
- 官方 DINOv2 Torch Hub 模型

检查环境：

~~~powershell
.\.venv\Scripts\python.exe -c "import torch, ultralytics; print(torch.__version__); print(torch.cuda.is_available()); print(ultralytics.__version__)"
~~~

代码默认从仓库根目录运行。

当前“YOLO剪枝蒸馏初步尝试”副本中的旧 `.venv` 仍指向已经不存在的
Windows Python，不能作为可复现环境。正式COCO2017实验使用新的服务器环境；
在确认服务器镜像和CUDA版本后再生成锁定依赖，避免沿用失效虚拟环境。

## 数据和基础权重

### 数据

| 数据集 | 用途 | 位置 |
| --- | --- | --- |
| COCO8 | 最小流程验证 | `datasets/coco8/` |
| COCO128 | 快速方法实验 | `datasets/coco128/` |
| COCO128 固定划分 | 102 train / 26 val | `datasets/coco128_split/` |
| COCO2017 | 正式大数据实验 | `C:/Users/22565/datasets/coco` |

COCO128 划分命令：

~~~powershell
.\.venv\Scripts\python.exe scripts\split_coco128.py
~~~

### 基础权重

| 路径 | 用途 |
| --- | --- |
| `models/yolo11n.pt` | COCO8 / COCO128 起始模型 |
| `weights/yolo11s.pt` | COCO2017 官方预训练基线 |
| `weights/yolo26n.pt` | 新模型变体探索 |

当前不存在 `weights/yolo11n.pt`，脚本和文档应使用真实路径。

## 剪枝实验

### 方法

- **敏感度分析**：逐层屏蔽 L1 最小的约 10% 通道，测量 mAP50-95 下降。
- **独立法**：light / balanced / strong 从同一起点分别剪枝。
- **贪心法**：每步试剪候选层，以 mAP 代价 / GMAC 收益选择当前最优步骤。
- **梯度分档**：按 L1 敏感度分层，用 `mean(|W × ∂L/∂W|)` 选择通道。
- **结构一致性**：Torch-Pruning DependencyGraph 联动修改下游 BN、卷积输入和相关深度卷积。

### COCO128 主要结果

| 模型 | 参数减少 | GMACs 减少 | mAP50-95 |
| --- | ---: | ---: | ---: |
| 未剪枝对照 | 0.00% | 0.00% | 0.6232 |
| 实验 04 independent strong（同设置复跑） | 4.17% | 4.07% | 0.5183 |
| 实验 05 greedy | 1.69% | **4.55%** | **0.5962** |
| 实验 06 gradient tier A | 4.17% | 4.07% | 0.5470 |

当前实验 05 是较好的压缩/精度折中，但相对未剪枝仍下降 0.0270。统一测速差异较小，尚无稳定端到端加速证据。

当前剪枝模型：

`runs/prune/experiment05_greedy/20260908_161152/finetune/greedy/weights/best.pt`

## 剪枝后的 DINOv2 蒸馏实验

### 方法

剪枝完成后，使用冻结的 DINOv2 ViT-S/14 作为教师，实验 05 的贪心剪枝模型作为学生：

$$
L_{total}=L_{YOLO}+\lambda(t)L_{DINO}
$$

- 实验 12：DINOv2 最终 patch tokens 对齐 YOLO P4；
- 实验 13：DINOv2 Block 4/8/12 对齐 YOLO P3/P4/P5；
- 使用独立 1×1 projector 解决通道差异；
- 教师特征插值到学生空间尺寸；
- 使用 cosine distance；
- 总蒸馏权重 0.5，前 2 epochs warm-up；
- 教师和 BatchNorm 冻结；
- 部署权重删除教师、projector 和 hook。

### 结果

| 方案 | best mAP50-95 | last mAP50-95 |
| --- | ---: | ---: |
| 输入剪枝模型 | 0.5962 | — |
| 同设置普通微调 | 0.5962 | 0.4828 |
| 实验 12：P4 单尺度 | 0.5962 | **0.5040** |
| 实验 13：P3/P4/P5 多尺度 | 0.5962 | 0.4957 |

两种蒸馏均未刷新 best。last 对照说明蒸馏缓解了后期退化，其中 P4 单尺度优于当前多尺度。

这不是“训练没有发生”：蒸馏损失下降，last 权重也发生变化。可能限制包括数据量太小、背景特征干扰、DINOv2 与检测任务不完全匹配，以及 projector 吸收部分特征对齐。

### 运行命令

P4 单尺度：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.distill_dinov2_coco128 `
  --epochs 30 `
  --batch 8 `
  --device 0 `
  --weight 0.5
~~~

P3/P4/P5 多尺度：

~~~powershell
.\.venv\Scripts\python.exe -m scripts.distill_dinov2_coco128 `
  --multiscale `
  --epochs 30 `
  --batch 8 `
  --device 0 `
  --weight 0.5
~~~

Linux Bash 使用 `.venv/bin/python`，续行符使用反斜杠。

## 目录结构

~~~text
YOLO/
├── configs/                 # COCO128 / COCO2017 数据配置
├── datasets/                # 仓库内小型数据集
├── Data/                    # 示例和自定义数据
├── models/                  # YOLO11n 权重
├── weights/                 # YOLO11s / YOLO26n 权重
├── scripts/                 # 训练、剪枝、蒸馏实现
├── runs/
│   ├── train/               # 训练结果
│   ├── baseline/            # 基线验证
│   ├── prune/               # 剪枝与微调
│   └── distill/             # 蒸馏正式实验与开发期 smoke
├── reports/                 # 正式 Markdown 报告与 CSV
├── Outputs/                 # 手动保留的展示输出
└── Summary & Learning/      # README 和三个学习 notebook
~~~

## 正式结果与开发产物

建议长期保留：

- `scripts/`、`configs/`、`reports/`；
- `Summary & Learning/`；
- 实验 05 正式剪枝目录；
- 实验 12、13 正式蒸馏目录；
- 每次正式运行的 `run_info.json`、`comparison.csv`、`results.csv` 和权重。

开发期 `smoke*` 目录仅用于检查设备、序列化、目录结构和小批量训练，不应当作正式科研结果。

## 实验边界

- COCO128 只有 102 张训练图和 26 张验证图；
- 同一验证集参与选层、选 best 和方法比较；
- 官方 COCO 预训练模型可能见过 COCO128 中的图片；
- 小样本负结果不能证明某种剪枝或蒸馏方法普遍无效；
- 速度结论必须统一设备、精度、输入、预热和测量方式；
- 正式科研结论应在完整 COCO2017、多个随机种子和统一对照上复核。

## 下一步
