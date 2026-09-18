# 实验19 最终对比：剪枝阶梯 × 教师蒸馏恢复

> 新增文件（不覆盖任何既有报告）。生成时间：2026-09-18 约 19:20 服务器时间。
> 数据来源：19A（结构化剪枝阶梯 + BN 校准）、19B（control + YOLO11s KD）、19C（YOLO11m KD + DINOv2 KD）。
> 全部训练：20 epoch `standard_recovery`（AdamW lr 1e-3 / warmup 1 / 开增强 / wd 5e-4 / 有效 batch 128）。

## 1. 教师蒸馏对比（核心结果）

19B Pareto 选出的两个节点（**均衡 = node15**、**激进 = node20**），19C 各跑 YOLO11m KD 与 DINOv2 KD，与 19B 的 control / YOLO11s KD 合并对比。val2017 mAP50-95：

| 节点 | GMAC | params | control | YOLO11s KD | YOLO11m KD | DINOv2 KD | 最佳 |
|---|---:|---:|---:|---:|---:|---:|---|
| 15（均衡） | 9.218 | 8,063,896 | 0.4235 | **0.4274** | 0.4246 | 0.4240 | 0.4274（yolo11s） |
| 20（激进） | 8.684 | 7,474,088 | 0.4215 | **0.4226** | 0.4217 | 0.4212 | 0.4226（yolo11s） |

**结论：同架构教师（YOLO11s）蒸馏最优；更大教师（YOLO11m）与跨模态教师（DINOv2 ViT-S）无额外增益。**

- YOLO11s KD：两节点一致小幅提升（node15 +0.0039，node20 +0.0011）。
- YOLO11m KD：几乎等于 control（node15 +0.0011，node20 +0.0002），更大的同族教师没有带来更多收益。
- DINOv2 KD：持平甚至略差（node15 +0.0005，node20 −0.0003）。

原因推测：学生在 15%/20% 剪枝后容量已很小（8–9 GMAC、8M 参数），难以吸收更强教师的特征；且 P3/P4/P5 余弦距离特征蒸馏（1×1 投影）与 YOLO 检测头在此规模下匹配度有限。

## 2. 剪枝阶梯：GMAC / params / 延迟

| 模型 | GMAC | GMAC↓ | params | params↓ | batch=1 推理（mean ms） | median ms |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11s 官方 | 10.7991 | — | 9,458,752 | — | 7.57 | 7.48 |
| node15（15%） | 9.218 | −14.6% | 8,063,896 | −14.7% | 7.50 | 7.48 |
| node20（20%） | 8.684 | −19.6% | 7,474,088 | −21.0% | 7.54 | 7.49 |

**关键发现：剪枝显著降低 GMAC（−15%/−20%）与参数（−15%/−21%），但 A800 上 batch=1 推理延迟几乎不变（~7.5ms）。**

单图推理在 A800 上是显存带宽/内核启动延迟受限，而非浮点算力受限，因此 channel 剪枝省下的 FLOPs 不转化为 batch=1 提速。剪枝的延迟收益只有在**高 batch 吞吐**或**算力受限设备**上才会体现。

## 3. 完整阶梯回顾（19A → 19B → 19C）

| 节点 | GMAC | BN 校准后（无训练）tune | 恢复后最佳 val2017 | 最佳方案 |
|---|---:|---:|---:|---|
| 5% | 10.281 | 0.5228 | 0.4256（=source） | 不训练（source） |
| 10% | 9.771 | 0.4683 | 0.3867（=source） | 不训练（source） |
| 15% | 9.218 | 0.3112 | **0.4274** | YOLO11s KD |
| 20% | 8.684 | 0.1356 | **0.4226** | YOLO11s KD |

规律（与 19B 报告一致）：浅剪枝（5%/10%）BN 校准后精度尚可，恢复配方反而训坏；深剪枝（15%/20%）BN 校准后塌陷、恢复训练强力拉回。KD 相对 control 的优势随深度先增后减，最优蒸馏源始终是同架构 YOLO11s。

## 4. 选定 checkpoint（SHA256）

| 节点 | 方案 | 权重路径 | SHA256 |
|---|---|---|---|
| 15 | yolo11s KD | `runs/recovery/experiment19b_coco2017/node15_yolo11s_kd/20260917_101816/training/weights/best_clean.pt` | `2441f27a…` |
| 15 | yolo11m KD | `runs/recovery/experiment19b_coco2017/node15_yolo11m_kd/20260917_131427/training/weights/best_clean.pt` | `ca696770…` |
| 15 | dinov2 KD | `runs/recovery/experiment19c_coco2017/node15_dinov2_kd/20260917_155923/training/weights/best_clean.pt` | `0c9945aa…` |
| 20 | yolo11s KD | `runs/recovery/experiment19b_coco2017/node20_yolo11s_kd/20260917_105730/training/weights/best_clean.pt` | `7ed225d5…` |
| 20 | yolo11m KD | `runs/recovery/experiment19b_coco2017/node20_yolo11m_kd/20260917_131427/training/weights/best_clean.pt` | `8e3df1ea…` |
| 20 | dinov2 KD | `runs/recovery/experiment19c_coco2017/node20_dinov2_kd/20260917_155447/training/weights/best_clean.pt` | `c4fe8d80…` |

（完整 SHA256 见各 run 目录 `comparison.csv`；此处前 8 位。control 的选定权重为各 `node*_control/.../best.pt`。）

## 5. 备注

- 19C 的 YOLO11m KD 结果因脚本 run 目录硬编码为 `experiment19b_coco2017`，实际落在 `experiment19b_coco2017/node{15,20}_yolo11m_kd/20260917_131427`；与 19B 的 yolo11s 目录不冲突、不覆盖。DINOv2 结果落在正确的 `experiment19c_coco2017/node{15,20}_dinov2_kd`。
- 19B control 任务曾因脚本 `write_report` KeyError bug 在写完 comparison.csv 后崩溃，已用 `finalize_control_report.py` 从 csv 重建 report/run_info，数据无损、未重跑。
- 延迟测量脚本：`/root/YOLO/exp19c_latency.py`（batch=1、imgsz=640、A800、200 次计时取 mean/median，仅模型前向，不含 NMS）。
