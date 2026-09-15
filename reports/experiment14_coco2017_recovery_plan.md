# 实验14：COCO2017模型谱系与恢复诊断计划

## 当前结论

本阶段先解决实验10的同源对照和恢复训练问题，不覆盖实验07–10，也不把旧的负结果改写为成功结果。

项目中存在两套YOLO11s基线：

| 权重 | SHA256 | 已记录mAP50-95 | 状态 |
|---|---|---:|---|
| `weights/yolo11s.pt` | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` | 0.4635 | 官方权重，实验07本地副本哈希一致 |
| `weights/baseline/yolo11s_coco2017_best.pt` | `336469c5bf22e9e2f53a033dfdf821e82b70a21fb13a128967013b074255b5b1` | 0.4450 | 实验09、10的父权重，本地缺失 |

实验10的 `pruned_raw.pt` SHA256为
`79b69ba5f14317e4d3957448601135f25ab55c1772625e9fae1f87cc102406f0`。
它从第二套权重剪出，不能与官方 `yolo11s.pt`直接构成同源剪枝对照。

## 决策

1. 实验10保留为负结果和工程记录。
2. 如果云端仍保留SHA256为`336469...`的父权重，可运行恢复诊断，判断BN策略和微调设置。
3. 正式COCO2017主线重新以官方`yolo11s.pt`为唯一父权重，重新计算Taylor分数并结构化剪枝。
4. 每个剪枝产物必须记录父权重SHA256；恢复脚本在训练前强制校验。
5. 训练后的best/last都与输入权重重新验证，最终选择范围包含输入权重，避免“best.pt比训练前更差”仍被保留。

## 恢复诊断设计

新增脚本：`scripts/diagnose_coco2017_recovery.py`。

三套预设的用途：

| 预设 | 与实验10的关系 | 用途 |
|---|---|---|
| `legacy_frozen` | 完全复现实验10的冻结BN和训练超参数 | 仅用于复现 |
| `bn_update` | 只把BN改为正常更新 | 第一项因果诊断 |
| `standard_recovery` | BN更新、提高学习率并恢复适度增强 | BN-only仍失败后再试 |

正式对照必须同时训练父模型和剪枝模型，使用相同数据、epoch、batch、优化器与增强。父模型同条件也下降时，不能把下降归因于剪枝。

## 运行顺序

### 仅做预检，不运行验证或训练

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_coco2017_recovery.py `
  --parent weights\baseline\yolo11s_coco2017_best.pt `
  --dataset-root C:\Users\22565\datasets\coco
```

### GPU服务器上执行5轮BN诊断

```bash
python scripts/diagnose_coco2017_recovery.py \
  --parent weights/baseline/yolo11s_coco2017_best.pt \
  --pruned runs/prune/experiment10_coco2017_greedy/20260909_133307/pruned_raw.pt \
  --provenance runs/prune/experiment10_coco2017_greedy/20260909_133307/run_info.json \
  --dataset-root /path/to/coco \
  --preset bn_update \
  --epochs 5 \
  --batch 32 \
  --val-batch 64 \
  --device 0 \
  --execute
```

服务器路径和batch需要根据实际租用实例填写。完整COCO验证和训练开始前再确定，不提前假设。

## 后续主线

```text
官方YOLO11s统一验证
-> 官方权重上重新统计Taylor分数
-> 固定10% GMAC预算进行迭代结构化剪枝
-> 父模型/剪枝模型同条件恢复训练
-> L1、Taylor、SFP同预算比较
-> Taylor模型接受普通微调
-> 原始YOLO11s教师蒸馏
-> DINOv2扩展对照
-> PyTorch/ONNX/TensorRT部署评测
```

当前阶段没有启动完整COCO2017验证、训练或剪枝，没有产生新的性能结论。

## 正式Taylor入口

新增 `scripts/prune_taylor_coco2017.py`，正式主线固定使用SHA256为
`85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`
的官方YOLO11s权重。默认流程为：

1. 从完整train2017中按seed固定抽取2048张校准图片；
2. FP32、BN不更新、不更新权重，统计每通道一阶Taylor代价；
3. 不使用val2017选择逐步剪枝动作；
4. 每次删除最多8个根通道，由Torch-Pruning传播结构依赖；
5. 按根通道Taylor代价除以实际GMAC收益选择动作；
6. 达到10% GMAC削减或没有可行动作后停止；
7. 完成前向、反向、保存重载校验；
8. 只对父模型和最终raw剪枝模型运行完整val2017。

预检命令不会运行校准、验证或剪枝：

```powershell
.\.venv\Scripts\python.exe scripts\prune_taylor_coco2017.py `
  --dataset-root C:\Users\22565\datasets\coco
```

服务器正式命令需要确认GPU、CUDA、数据盘路径和batch后再填写。环境依赖入口为
`requirements-coco2017.txt`；PyTorch根据服务器CUDA镜像单独确定。
