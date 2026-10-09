# YOLO 剪枝与 DINOv2 蒸馏学习资料

这里集中保存本项目的学习笔记与结构说明，目标是把“实验怎么做、代码为什么这样写、结果应该怎样解释”整理成可复习、可答辩、可继续科研的材料。

当前内容覆盖 YOLO11 在 COCO8、COCO128、COCO2017 上的训练、结构化剪枝，以及以 DINOv2 ViT-S/14 为教师的单尺度和多尺度特征蒸馏。

## 阅读导航

建议按以下顺序阅读：

| 顺序 | 文件 | 主要内容 | 适合解决的问题 |
| ---: | --- | --- | --- |
| 1 | [项目结构与数据流讲解.ipynb](./项目结构与数据流讲解.ipynb) | 目录职责、数据来源、脚本输入输出、端到端数据流 | “项目里的东西都在哪，前后怎样连接？” |
| 2 | [01pruning.ipynb](./01pruning.ipynb) | 结构化剪枝、敏感度、独立法、贪心法、梯度分档、测速 | “剪枝原理是什么，关键代码怎样写？” |
| 3 | [02distillation.ipynb](./02distillation.ipynb) | 特征蒸馏、单/多尺度、损失、命令、结果解释 | “蒸馏怎样接入 YOLO？” |
| 4 | [03coco2017.ipynb](./03coco2017.ipynb) | 完整数据集上的剪枝、恢复、蒸馏和比较主线 | “正式实验从输入到结果怎样连接？” |
| 5 | [04完整项目终端与代码框架.md](./04完整项目终端与代码框架.md) | 从 Git 获取到训练、剪枝、蒸馏、报告和 push 的通用模板 | “拿到新项目后，终端和代码按什么顺序写？” |

根目录的总体项目说明见 [../README.md](../README.md)，各次实验的正式结果见 [../reports/](../reports/)。

## 项目路线

| 阶段 | 数据 + 模型 | 实验 | 工作内容 |
| --- | --- | --- | --- |
| 1 | COCO8 + YOLO11n | 01 | 环境和训练/验证流程验证 |
| 2 | COCO128 + YOLO11n | 02–06 | 基线训练、逐层敏感度、独立剪枝、贪心剪枝、梯度分档剪枝 |
| 3 | COCO2017 + YOLO11s | 07–10 | 官方基线、训练尝试、敏感度分析、贪心剪枝 |
| 4 | COCO128 + 剪枝 YOLO11n + DINOv2 | 12–13 | P4 单尺度特征蒸馏、P3/P4/P5 多尺度特征蒸馏 |

实验 11 主要是结构化剪枝方法与代码对照资料，并包含 GroupNorm 等探索，不作为与实验 01–10 相同口径的正式性能结论。

## 文件内容

### 项目结构与数据流

[项目结构与数据流讲解.ipynb](./项目结构与数据流讲解.ipynb) 负责解释：

- `configs/`、`datasets/`、`models/`、`weights/`、`scripts/`、`runs/`、`reports/` 的职责；
- COCO8、COCO128、COCO2017 的来源、规模和存放位置；
- 每个训练、剪枝、蒸馏脚本的主要输入和输出；
- `.pt` 检查点、`best.pt`、`last.pt`、EMA 和训练恢复信息；
- 从数据到训练、剪枝、蒸馏、验证、报告的完整流向。

### 剪枝学习笔记

[01pruning.ipynb](./01pruning.ipynb) 按“原理 → Bash 命令 → 关键代码”讲解：

- 结构化剪枝与非结构化剪枝的区别；
- L1 masking 敏感度分析；
- Torch-Pruning DependencyGraph；
- 独立法、贪心法和梯度分档剪枝；
- `mean(|W × ∂L/∂W|)` 梯度重要性；
- 冻结 BatchNorm 的剪枝后恢复训练；
- GMACs、参数量、mAP50-95 和 CUDA 前向测速口径。

### 蒸馏学习笔记

[02distillation.ipynb](./02distillation.ipynb) 按同样格式讲解：

- 知识蒸馏、输出蒸馏和特征蒸馏；
- DINOv2 ViT-S/14 为什么使用 644×644 教师输入；
- YOLO P3/P4/P5 与 DINO patch token 的空间、通道对齐；
- 1×1 projector、cosine distance、蒸馏权重和 warm-up；
- forward hook、自定义 YOLO loss、冻结教师与 BN；
- PowerShell 和 Linux Bash 的完整运行命令；
- 训练 wrapper 与纯 YOLO 部署权重的区别；
- 实验 12、13 的真实结果、局限、排错清单和老师问答。

## 主要实验结论

### 剪枝

COCO128 阶段当前建议保留实验 05 的贪心剪枝模型：

| 模型 | GMACs 减少 | mAP50-95 | 相对未剪枝下降 |
| --- | ---: | ---: | ---: |
| 未剪枝对照 | 0.00% | 0.6232 | 0.0000 |
| 实验 04 独立法 strong（同设置复跑） | 4.07% | 0.5183 | -0.1049 |
| 实验 05 贪心法 | **4.55%** | **0.5962** | -0.0270 |

贪心法在相近计算量下降下明显优于独立法，但参数仅减少约 1.69%，实测速度差异也小于波动，不能宣称已经获得稳定端到端加速。

实验 06 的梯度分档方案最高为 0.5470，未达到预设的 mAP50-95 损失不超过 0.02 的要求。因此现阶段剪枝结果用于方法验证，仍需更大数据集复核。

### DINOv2 蒸馏

学生为实验 05 贪心剪枝后的 `best.pt`，教师为冻结的 DINOv2 ViT-S/14。

| 方案 | best mAP50-95 | last mAP50-95 |
| --- | ---: | ---: |
| 输入剪枝模型 | 0.5962 | — |
| 同设置普通微调 | 0.5962 | 0.4828 |
| 实验 12：P4 单尺度 DINOv2 | 0.5962 | **0.5040** |
| 实验 13：P3/P4/P5 多尺度 DINOv2 | 0.5962 | 0.4957 |

阶段结论：

- 单尺度、多尺度蒸馏均正常参与反向传播；
- 两种蒸馏都没有刷新输入模型的峰值精度；
- 蒸馏 last 优于普通微调 last，说明它缓解了部分后期退化；
- 当前 P4 单尺度优于多尺度；
- COCO128 只有 102 张训练图和 26 张验证图，不能据此断言 DINOv2 蒸馏普遍无效；
- 下一阶段应在完整 COCO2017 上优先比较“无蒸馏 vs P4 单尺度”，而不是继续在 COCO128 上大量调参。

## 重要命令

以下命令默认从仓库根目录执行。

### 启动学习笔记

PowerShell：

~~~powershell
.\.venv\Scripts\python.exe -m jupyter lab "Summary & Learning"
~~~

Linux Bash：

~~~bash
.venv/bin/python -m jupyter lab "Summary & Learning"
~~~

如果环境没有安装 Jupyter，可直接在 VS Code 中打开三个 `.ipynb`。

### COCO128 数据划分

~~~powershell
.\.venv\Scripts\python.exe scripts\split_coco128.py
~~~

### P4 单尺度 DINOv2 蒸馏

~~~powershell
.\.venv\Scripts\python.exe -m scripts.distill_dinov2_coco128 `
  --epochs 30 `
  --batch 8 `
  --device 0 `
  --weight 0.5
~~~

### P3/P4/P5 多尺度 DINOv2 蒸馏

~~~powershell
.\.venv\Scripts\python.exe -m scripts.distill_dinov2_coco128 `
  --multiscale `
  --epochs 30 `
  --batch 8 `
  --device 0 `
  --weight 0.5
~~~

Linux Bash 使用 `.venv/bin/python`，并把 PowerShell 续行反引号改为反斜杠 `\`。

## 关键输入与输出

~~~text
官方 YOLO 权重 + 数据配置
            │
            ├──→ train / baseline ──→ runs/train、runs/baseline
            │
            ├──→ 敏感度分析 ──→ reports/*sensitivity.csv
            │
            ├──→ 结构化剪枝 + 微调 ──→ runs/prune/
            │                              │
            │                              └──→ 实验05 greedy best.pt
            │                                           │
            └───────────────────────────────────────────┴──→ DINOv2 蒸馏
                                                             │
                                                             ├──→ runs/distill/
                                                             └──→ reports/experiment12、13
~~~

常用位置：

- 数据配置：`../configs/coco128_split.yaml`、`../configs/coco2017.yaml`
- 实验脚本：`../scripts/`
- 剪枝产物：`../runs/prune/`
- 蒸馏产物：`../runs/distill/`
- 人工可读报告：`../reports/`
- 当前蒸馏学生：`../runs/prune/experiment05_greedy/20260908_161152/finetune/greedy/weights/best.pt`

## 阅读和复现实验时的注意事项

- Notebook 中的代码以讲解关键逻辑为主，不保证单独一个单元即可完整运行；正式入口始终是 `scripts/` 中的脚本。
- `best.pt` 表示按验证指标选择的检查点，`last.pt` 表示最后一轮；二者必须同时理解。
- 蒸馏训练 wrapper 含教师和 projector，部署 `best.pt` / `last.pt` 已删除这些训练组件。
- COCO128 验证集还参与了选层、选方案和选 best，结果属于开发集证据。
- 推理耗时必须固定设备、精度、输入、预热和测量次数；不能直接比较不同验证调用打印的单次速度。
- 对正式科研结论，应使用完整 COCO2017、独立验证、多个随机种子和统一对照。

## 校验状态

截至 2026-09-10：

- 三个 notebook 均为有效 nbformat 4 JSON；
- 共 95 个单元，无空单元、无保存的执行输出；
- notebook 之间的相对链接有效；
- 蒸馏笔记已删除“计划阶段/尚未实现”占位内容；
- 项目结构说明已同步实验 12、13 和当前真实权重位置；
- 蒸馏关键数值已与正式报告核对；
- 学习笔记只记录讲解内容，不复制大型权重和运行产物。

## 后续更新规则

新增实验时建议同时更新：

1. `reports/experimentXX_*.md`：正式结论和数值；
2. 对应的剪枝或蒸馏 notebook：原理、命令和关键代码；
3. `项目结构与数据流讲解.ipynb`：新增脚本输入输出；
4. 本 README：实验路线、主要结论和阅读导航。

这样可以避免“脚本已经完成，但学习资料仍写着计划阶段”的不一致。

