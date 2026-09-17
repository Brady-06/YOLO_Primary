#!/bin/bash
# 实验19C 调度器：对 19B Pareto 选出的 2 个节点，各跑 YOLO11m KD + DINOv2 KD。
# 用法：bash scripts/run_experiment19c.sh <均衡节点> <激进节点>
#   例：bash scripts/run_experiment19c.sh 10 20
# 车道分配（每节点 YOLO11m KD ~2h，DINOv2 KD ~2.2h）：
#   lane0(GPU0): 均衡节点 YOLO11m KD -> 均衡节点 DINOv2 KD
#   lane1(GPU1): 激进节点 YOLO11m KD -> 激进节点 DINOv2 KD
set -u
cd /root/YOLO
PY=/usr/local/miniconda3/envs/py312/bin/python
DATA=/root/datasets/coco
MASTER=exp19c_scheduler.log
if [ $# -ne 2 ]; then
  echo "usage: $0 <node_balanced> <node_aggressive>" >&2
  exit 2
fi
NODE_BAL=$1
NODE_AGG=$2

case "$NODE_BAL" in
  5) SRC_BAL=runs/recovery/experiment19_bn_recalibration_coco2017/20260915_133506/preserve_m0005_8192.pt ;;
  10) SRC_BAL=runs/prune/experiment20_taylor_ladder_coco2017/20260917_023832/stage_10/bn_m0005_8192.pt ;;
  15) SRC_BAL=runs/prune/experiment20_taylor_ladder_coco2017/20260917_023832/stage_15/bn_m0005_8192.pt ;;
  20) SRC_BAL=runs/prune/experiment20_taylor_ladder_coco2017/20260917_023832/stage_20/bn_m0005_8192.pt ;;
  *) echo "unknown balanced node $NODE_BAL" >&2; exit 2 ;;
esac
case "$NODE_AGG" in
  5) SRC_AGG=runs/recovery/experiment19_bn_recalibration_coco2017/20260915_133506/preserve_m0005_8192.pt ;;
  10) SRC_AGG=runs/prune/experiment20_taylor_ladder_coco2017/20260917_023832/stage_10/bn_m0005_8192.pt ;;
  15) SRC_AGG=runs/prune/experiment20_taylor_ladder_coco2017/20260917_023832/stage_15/bn_m0005_8192.pt ;;
  20) SRC_AGG=runs/prune/experiment20_taylor_ladder_coco2017/20260917_023832/stage_20/bn_m0005_8192.pt ;;
  *) echo "unknown aggressive node $NODE_AGG" >&2; exit 2 ;;
esac

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_yolo11m_kd() { # node source gpu
  local node=$1 src=$2 gpu=$3
  log "START yolo11m_kd node=$node gpu=$gpu"
  $PY scripts/recover_ladder_yolo_kd_coco2017.py --node "$node" --source "$src" \
    --teacher yolo11m.pt --teacher-name yolo11m --dataset-root "$DATA" \
    --execute --device "$gpu" --epochs 20 --batch 128 --nbs 128 \
    >> "exp19c_node${node}_yolo11m_kd.log" 2>&1
  log "END   yolo11m_kd node=$node gpu=$gpu rc=$?"
}

run_dinov2_kd() { # node source gpu
  local node=$1 src=$2 gpu=$3
  log "START dinov2_kd node=$node gpu=$gpu"
  $PY scripts/recover_ladder_dinov2_kd_coco2017.py --node "$node" --source "$src" \
    --dataset-root "$DATA" --execute --device "$gpu" --epochs 20 --batch 128 --nbs 128 \
    --kd-weight 0.5 --kd-warmup 2 --dino-size 644 \
    >> "exp19c_node${node}_dinov2_kd.log" 2>&1
  log "END   dinov2_kd node=$node gpu=$gpu rc=$?"
}

lane0() {
  run_yolo11m_kd "$NODE_BAL" "$SRC_BAL" 0
  run_dinov2_kd "$NODE_BAL" "$SRC_BAL" 0
  log "lane0: DONE (balanced node $NODE_BAL)"
}

lane1() {
  run_yolo11m_kd "$NODE_AGG" "$SRC_AGG" 1
  run_dinov2_kd "$NODE_AGG" "$SRC_AGG" 1
  log "lane1: DONE (aggressive node $NODE_AGG)"
}

log "SCHEDULER START balanced=$NODE_BAL aggressive=$NODE_AGG"
lane0 >> "$MASTER" 2>&1 &
lane1 >> "$MASTER" 2>&1 &
wait
log "SCHEDULER ALL DONE"
