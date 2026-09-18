# 实验19最终对比：剪枝阶梯与蒸馏恢复（服务器数据核验版）

> 核验时间：2026-09-18T10:19:35+08:00（北京时间）。
> 来源：用户指定服务器 `/root/YOLO` 中的原始 run_info、comparison.csv、training/results.csv、args.yaml 和实际权重。
> 本次没有重新训练或重新计算COCO精度；复核了文件哈希、模型结构、两种GMAC计数，并运行了已有的短延迟测试。

## 1. 主要结论

19A完成了约5%、10%、15%、20%的raw嵌套剪枝；19B/19C共12个正式训练任务均完成20轮。单轮测试、预检和失败尝试已排除出正式比较。

- **当前val最高的压缩候选：node15 + YOLO11s KD，mAP50-95 = 0.427374**，TP口径GMAC减少15.13%，参数减少14.75%。
- **计算量更低的候选：node20 + YOLO11s KD，mAP50-95 = 0.422572**，TP口径GMAC减少20.06%，参数减少20.98%。
- 深剪枝的大部分恢复来自普通检测训练。父模型蒸馏在node15/node20的val上分别额外增加约 **0.39/0.10 AP点**；只是一次seed=42实验的观察，不能称为已证明稳定增益。
- 相对历史官方基线约0.4633，两候选仍低约3.59/4.07 AP点。没有实现无损压缩，也没有在A800单图前向测试中证实明显加速。
- YOLO11s在本次两个节点的val上领先；tune上的教师排序不同。node15/node20与教师优劣使用了val作事后比较，因此这些val结果不是完全独立于选型的最终测试成绩。

数值约定：mAP保留0～1范围；“AP点”按mAP差值×100计算。恢复前后只在同一数据集内相减。

## 2. 与计划的符合程度

| 项目 | 核验结果 |
|---|---|
| 官方权重、5% raw起点及嵌套链路 | run_info记录与方案一致；后续从raw继续，未用训练权重续剪 |
| 固定数据隔离 | calibration=2048、tune=2048、recovery=114191；按实际图片文件名核查，两两交集均为0 |
| 8通道粒度、Taylor/GMAC贪心 | 96个后续接受动作；每步重算Taylor，代码按比值择优；不是重新独立搜索四个节点 |
| 层级保护 | 保留每档low/medium/high上限；**没有实现计划中逐级开放**，每档直接纳入所有上限内候选 |
| BN与raw隔离 | 统一8192张、momentum=0.005、不重置；在副本上校准，不将BN权重送回主剪枝链 |
| BN执行时机 | **实际每到节点就做BN及评价**，而非全部剪完才校准；主raw链未被校准更新 |
| 四节点control/YOLO11s，两个节点YOLO11m/DINOv2 | 8+4个正式任务完成；源权重哈希一致 |
| 公平训练预算 | 12份args.yaml关键参数一致，12份results.csv均为20轮 |
| source/best/last选择 | CSV均按tune最大值选择，再对选中权重做val；5%/10%均选回source |
| 部署权重 | 13份基线/source/选中权重完成CPU加载与双计数；选中学生与source形状及参数量一致，无教师或投影模块残留 |
| 精度与效率最终评价 | 已有mAP和短前向延迟；未测试端到端、目标端设备或多次独立重复 |
| 5%配方验证 | 有一次单轮KD测试，但训练后tune很低且正式5%仍退化；不能把“代码跑通”写成“恢复配方已证实有效” |

结论：主体实验已完成，但上述执行偏差和评价限制必须保留。不会为了让结果看似符合而追改原计划。

## 3. 数据、环境和固定配方

训练记录：Python 3.12.11、PyTorch 2.13.0+cu132、Ultralytics 8.4.144、A800-SXM4-80GB；本次计数环境Torch-Pruning 1.6.1、ultralytics-thop 2.1.6。正式记录没有完整训练时Git commit及所有库版本快照，本次环境版本不能冒充历史版本锁定。

训练统一：20 epochs、imgsz=640、物理batch=128、nbs=128、AdamW、lr0=0.001、lrf=0.1、warmup_epochs=1、warmup_bias_lr=0.01、weight_decay=0.0005、AMP、seed=42、deterministic=True。增强为mosaic=0.5、fliplr=0.5、scale=0.3、translate=0.1、HSV=0.015/0.7/0.4，close_mosaic=10。此处不需要靠梯度累积把不同物理batch凑齐。

YOLO教师使用冻结的P3/P4/P5特征与训练期1×1投影、余弦距离。DINOv2为**跨架构视觉教师**（不是跨模态），型号ViT-S/14，blocks 4/8/12对应P3/P4/P5；输入插值至644、按DINO均值方差归一化，投影至384通道；层权重0.25/0.50/0.25。KD总权重0.5、2轮线性升权。教师均冻结，部署时移除教师和投影。

BN额外使用recovery池中的8192张，而非仅2048张Taylor校准集；不使用tune标签做更新。数据清单摘要及交集结果见审计JSON；原始数据yaml也已留档。

## 4. 统一计算量口径

本次把官方、source和正式选中学生全部用同一环境、CPU FP32、eval、输入1×3×640×640重测。主口径采用计划原来的Torch-Pruning；THOP作为并列复核，绝不跨工具计算比例。

| 模型 | 参数量 | TP GMAC | TP降幅 | THOP GMAC | THOP降幅 |
|---|---:|---:|---:|---:|---:|
| 官方YOLO11s | 9,458,752 | 10.7990592 | 0.00% | 10.8567296 | 0.00% |
| node5 | 9,093,536 | 10.2250816 | 5.32% | 10.2810624 | 5.30% |
| node10 | 8,549,624 | 9.7159040 | 10.03% | 9.7706880 | 10.00% |
| node15 | 8,063,896 | 9.1649568 | 15.13% | 9.2184224 | 15.09% |
| node20 | 7,474,088 | 8.6322592 | 20.06% | 8.6844000 | 20.01% |

旧报告把TP官方10.7991与THOP学生9.218/8.684混算；现已纠正。THOP官方实际为10.8567296。两工具计数不同不代表训练改变结构：每个入选学生与其source在两种计数下均一致。

## 5. BN收益与训练前基线

| 节点 | raw tune | BN/source tune | BN提升（tune） | source val |
|---:|---:|---:|---:|---:|
| 5 | 0.514366 | 0.522828 | +0.008462 | 0.425604 |
| 10 | 0.431151 | 0.468309 | +0.037159 | 0.386686 |
| 15 | 0.195039 | 0.311249 | +0.116210 | 0.258892 |
| 20 | 0.053326 | 0.135618 | +0.082292 | 0.117799 |

5% raw完整val约0.4181，BN后0.425604；10/15/20% raw只有tune结果，没有raw完整val，因此不虚构这三个节点的val校准收益。

## 6. 12个正式任务完整结果

“训练后最高tune”取保存的best/last中较高者；“选中tune/val”对应source/best/last最终胜出者。5%/10%选回source后的val不应冒充训练后模型的val。

| 节点 | 方案 | 训练后最高tune | 选中 | 选中tune | 选中val | val相对source | val相对control |
|---:|---|---:|---|---:|---:|---:|---:|
| 5 | control | 0.454161 | source | 0.522828 | 0.425604 | +0.000000 | +0.000000 |
| 5 | YOLO11s KD | 0.453540 | source | 0.522828 | 0.425604 | +0.000000 | +0.000000 |
| 10 | control | 0.449928 | source | 0.468309 | 0.386686 | +0.000000 | +0.000000 |
| 10 | YOLO11s KD | 0.456338 | source | 0.468309 | 0.386686 | +0.000000 | +0.000000 |
| 15 | control | 0.448279 | best | 0.448279 | 0.423467 | +0.164575 | +0.000000 |
| 15 | YOLO11s KD | 0.451473 | best | 0.451473 | 0.427374 | +0.168482 | +0.003907 |
| 15 | YOLO11m KD | 0.452120 | best | 0.452120 | 0.424587 | +0.165695 | +0.001120 |
| 15 | DINOv2 KD | 0.445610 | best | 0.445610 | 0.423965 | +0.165073 | +0.000498 |
| 20 | control | 0.449289 | best | 0.449289 | 0.421530 | +0.303731 | +0.000000 |
| 20 | YOLO11s KD | 0.443622 | best | 0.443622 | 0.422572 | +0.304773 | +0.001042 |
| 20 | YOLO11m KD | 0.448004 | best | 0.448004 | 0.421673 | +0.303874 | +0.000143 |
| 20 | DINOv2 KD | 0.446232 | best | 0.446232 | 0.421245 | +0.303447 | -0.000284 |

5%/10%完成了训练，但该20轮恢复配方没有超过source；应写“训练后按tune选回source”，不是“未训练”。未入选best/last没有完整val，不做跨数据集推断。

node15：source 0.258892 → control 0.423467（+0.164575）→ YOLO11s KD 0.427374（相对control +0.003907）。

node20：source 0.117799 → control 0.421530（+0.303731）→ YOLO11s KD 0.422572（相对control +0.001042）。

因此不能把深剪枝从0.12/0.26恢复至约0.42的全部收益归因于蒸馏。YOLO11m和DINOv2没有超过YOLO11s的val成绩，但YOLO11m两点、DINOv2在node15仍有很小的正差值；不写成“完全没有收益”。单种子且没有重复/置信区间，不证明这些小差异稳定或显著；学生容量不足、特征不匹配只是待验证假设。

## 7. 选型及适用范围

原19B用val–GMAC选出node15/node20：在该坐标下node15支配node5和node10。这仅说明这批已测权重的计算量与val关系，不能把GMAC较低称为“必然更快”。

**tune排序不同**：5% source为0.522828、10% source为0.468309；不能声称它们在tune上被node15淘汰。19C中node15的YOLO11m tune=0.452120，高于YOLO11s的0.451473；node20 control tune=0.449289，也高于YOLO11s的0.443622。

因此保留node15+YOLO11s作为“本次val表现最好的压缩候选”、node20+YOLO11s作为低GMAC候选，不把它们包装成严格仅依赖tune选出的普适最佳模型。若需要独立泛化结论，应冻结选型后另用独立测试数据验证。官方模型历史val约0.4633仍高于全部压缩候选；本文未重跑基线精度。

## 8. 延迟：历史值和本次复测

已有脚本 `exp19c_latency.py` 使用A800、batch=1、640×640、全零输入、eval/no_grad、30次预热、200次计时；每次前后CUDA同步，perf_counter墙钟时间。没有启用autocast或half，也没有显式融合、编译或TensorRT导出；仅模型前向，不含预处理、传输和NMS。模型加载后按默认FP32路径运行。

| 模型 | 历史mean ms（旧报告） | 历史median ms | 本次mean ms | 本次median ms | 本次P90 ms |
|---|---:|---:|---:|---:|---:|
| yolo11s_official | 7.57 | 7.48 | 7.86 | 7.80 | 7.90 |
| node15_15pct | 7.50 | 7.48 | 7.76 | 7.73 | 7.78 |
| node20_20pct | 7.54 | 7.49 | 7.99 | 7.92 | 8.10 |

本次测量前两张A800都显示0%利用率、0MiB显存占用。新旧结果都没有显示随15%/20% GMAC下降而出现相应的加速；20%本次反而略慢。不能据此确认访存或内核启动是哪一种瓶颈，也不能保证换成大batch/端侧一定提速。

旧报告数值作为历史转录保留，本次完整stdout另存；200次前向并非200次独立实验。5%/10%未补测同协议延迟；当前结论限定于官方和两个入选节点，不外推整个阶梯的实时性能。

## 9. 实际训练耗时与显存

耗时取training/results.csv最后一行累计time，包含该训练循环的验证开销，不等于纯GPU核时间；不包含后续独立checkpoint评价，也不把并行任务时长之和当墙钟总时间。显存为脚本记录的PyTorch峰值allocated（GiB），不是整卡占用。

| 节点 | 方案 | 轮数 | 累计训练时间h | 峰值allocated GiB |
|---:|---|---:|---:|---:|
| 5 | control | 20 | 1.446 | 未留存 |
| 5 | YOLO11s KD | 20 | 2.087 | 30.65 |
| 10 | control | 20 | 1.475 | 未留存 |
| 10 | YOLO11s KD | 20 | 2.094 | 30.69 |
| 15 | control | 20 | 1.445 | 未留存 |
| 15 | YOLO11s KD | 20 | 2.077 | 30.19 |
| 15 | YOLO11m KD | 20 | 2.705 | 33.12 |
| 15 | DINOv2 KD | 20 | 3.117 | 35.60 |
| 20 | control | 20 | 1.384 | 未留存 |
| 20 | YOLO11s KD | 20 | 2.011 | 29.03 |
| 20 | YOLO11m KD | 20 | 2.628 | 33.11 |
| 20 | DINOv2 KD | 20 | 3.041 | 35.02 |

control的run_info经事后补报告后缺少峰值显存与training_settings；本次已用实际args.yaml核对配方，不推算缺失的峰值。

## 10. 选中权重和复现记录

全部12个正式结果的完整路径、完整SHA256、大小、source哈希和结构摘要见[checkpoint_manifest.csv](experiment19_audit/20260918_101935/checkpoint_manifest.csv)。本次实际读取13个不同文件复算SHA，与原CSV逐一相符；所有选中学生的形状、参数量及双口径GMAC与source相同。文件大小是实际.pt容器字节数，含元数据，不等同于纯参数大小。

**node15 YOLO11s KD**（15.70 MiB）

```text
/root/YOLO/runs/recovery/experiment19b_coco2017/node15_yolo11s_kd/20260917_101816/training/weights/best_clean.pt
SHA256: 2441f27a86b5e04046b409a367b1df4cee730703f98b1778c85433ccf5601404
```

**node20 YOLO11s KD**（14.57 MiB）

```text
/root/YOLO/runs/recovery/experiment19b_coco2017/node20_yolo11s_kd/20260917_105730/training/weights/best_clean.pt
SHA256: 7ed225d564cd29ee96504a730f58d59be658772f426f2987c118ff7e017cad4a
```

教师权重记录（YOLO11s亦为官方剪枝起点）：

| 教师 | SHA256 |
|---|---|
| YOLO11s KD | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| YOLO11m KD | `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` |
| DINOv2 KD | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |

核验附件：

- [完整精度/资源CSV](experiment19_audit/20260918_101935/comparison_full.csv)：含mAP50、precision、recall、source/control差值、耗时和运行路径。
- [审计JSON](experiment19_audit/20260918_101935/audit.json)：实际文件哈希、大小、参数、TP/THOP计数、模型形状摘要、数据清单哈希与交集。
- [延迟复测stdout](experiment19_audit/20260918_101935/latency_recheck.txt)与[原延迟脚本](experiment19_audit/20260918_101935/server/exp19c_latency.py)。
- [修改前最终报告](experiment19_audit/20260918_101935/final_comparison_before.md)。
- 原始run_info、comparison.csv、args.yaml、results.csv及脚本按服务器相对路径保存在本附件的server目录。原始文件不改写，正式权重仍在服务器。

数据清单哈希口径：取各集合排序后的图片文件名，用换行连接后计算SHA256；不冒充图片内容哈希。数据yaml相同不等于内容相同，本次另核查了清单计数与交集。

## 11. 异常、缺口和未证实事项

- 4个control在写报告时曾出现KeyError，现run_info标记finalized_posthoc；results.csv和comparison.csv存在。正式任务均有20行训练数据，但事后补报告时间不能当作准确的原训练结束时间。
- YOLO11m任务目录仍叫experiment19b，按configuration.teacher识别为19C，未与YOLO11s混淆。
- 1轮KD测试和失败预检不加入12个正式结果；见下表。source/best/last原始记录均保留。
- 原19A记录中“19B/19C待续”为当时阶段快照；本核验报告为当前完成状态的总表，未回改历史记录。
- 层级逐步开放未实现；每个节点中途在副本上校准/评价；5%配方的恢复有效性没有被证明。不能标注为严格完全符合全部原计划条款。
- 模型精度没有由本次重新跑COCO确认，采用原始评估CSV；节点和教师val排名只是当前单种子结果。未重跑不同剪枝方法、不同种子、独立测试集或端侧部署。

| 排除运行（相对/root/YOLO） | 状态 | epochs |
|---|---|---:|
| `runs/recovery/experiment19b_coco2017/node5_control/20260917_050800` | preflight_complete | 20 |
| `runs/recovery/experiment19b_coco2017/node5_control/20260917_050847` | failed | 20 |
| `runs/recovery/experiment19b_coco2017/node5_yolo11s_kd/20260917_052604` | preflight_complete | 20 |
| `runs/recovery/experiment19b_coco2017/node5_yolo11s_kd/20260917_052718` | failed | 1 |
| `runs/recovery/experiment19b_coco2017/node5_yolo11s_kd/20260917_054107` | complete | 1 |

## 12. 实验回答

在这条Taylor嵌套路径上，约15%～20%的结构压缩可以经20轮训练恢复到约0.42 mAP，但没有恢复至官方约0.4633。普通微调贡献了主要恢复，YOLO11s特征蒸馏在本次val上带来小幅额外收益。参数和理论计算量的减少已经验证；A800 batch=1明显加速、教师普适最优、无损压缩仍未得到证实。
