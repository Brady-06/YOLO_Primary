#!/bin/bash
# 实验19B 调度器：2 GPU 车道并行，串行跑完剩余任务。
# 前提：node5 control 已在 GPU0 运行（本脚本 lane0 会等它结束）。
# 车道分配（control ~1.4h，KD ~2.0h）：
#   lane0(GPU0): node15 control -> node5 KD -> node15 KD
#   lane1(GPU1): node10 control -> node20 control -> node10 KD -> node20 KD
set -u
cd /root/YOLO
PY=/usr/local/miniconda3/envs/py312/bin/python
DATA=/root/datasets/coco
MASTER=exp19b_scheduler.log

SRC5=runs/recovery/experiment19_bn_recalibration_coco2017/20260915_133506/preserve_m0005_8192.pt
SRC10=runs/prune/experiment20_taylor_ladder_coco2017/20260917_023832/stage_10/bn_m0005_8192.pt
SRC15=runs/prune/experiment20_taylor_ladder_coco2017/20260917_023832/stage_15/bn_m0005_8192.pt
SRC20=runs/prune/experiment20_taylor_ladder_coco2017/20260917_023832/stage_20/bn_m0005_8192.pt

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_control() { # node source gpu
  local node=$1 src=$2 gpu=$3
  log "START control node=$node gpu=$gpu src=$src"
  $PY scripts/recover_ladder_control_coco2017.py --node "$node" --source "$src" \
    --dataset-root "$DATA" --execute --device "$gpu" --epochs 20 --batch 128 --nbs 128 \
    >> "exp19b_node${node}_control.log" 2>&1
  log "END   control node=$node gpu=$gpu rc=$?"
}

run_kd() { # node source gpu
  local node=$1 src=$2 gpu=$3
  log "START kd node=$node gpu=$gpu src=$src"
  $PY scripts/recover_ladder_yolo_kd_coco2017.py --node "$node" --source "$src" \
    --teacher weights/yolo11s.pt --teacher-name yolo11s --dataset-root "$DATA" \
    --execute --device "$gpu" --epochs 20 --batch 128 --nbs 128 \
    >> "exp19b_node${node}_kd.log" 2>&1
  log "END   kd node=$node gpu=$gpu rc=$?"
}

lane0() {
  while pgrep -f "recover_ladder_control_coco2017.py --node 5 " >/dev/null; do sleep 45; done
  log "lane0: node5 control finished"
  run_control 15 "$SRC15" 0
  run_kd 5 "$SRC5" 0
  run_kd 15 "$SRC15" 0
  log "lane0: DONE"
}

lane1() {
  run_control 10 "$SRC10" 1
  run_control 20 "$SRC20" 1
  run_kd 10 "$SRC10" 1
  run_kd 20 "$SRC20" 1
  log "lane1: DONE"
}

log "SCHEDULER START"
lane0 >> "$MASTER" 2>&1 &
lane1 >> "$MASTER" 2>&1 &
wait
log "SCHEDULER ALL DONE"
