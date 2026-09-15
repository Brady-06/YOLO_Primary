# Experiment 17: tiered Taylor pruning on COCO2017

- Source: official YOLO11s.
- Sensitivity tiers: 14 low, 13 medium, 8 high; high-sensitivity layers were locked.
- Search: dependency-aware greedy Taylor cost per actual GMAC saved; Taylor scores were recalibrated after every accepted action.
- Constraints: 8-channel steps, low-tier cap 12.5%, medium-tier cap 6.25%, and a 5% GMAC target.
- Result: 24 accepted actions; 10.7991 to 10.2251 GMAC (`-5.32%`).
- Parameters: 9,458,752 to 9,093,536 (`-3.86%`).
- Raw COCO val2017 mAP50-95: `0.4181` versus official source `0.4633`.
- Selected raw checkpoint SHA256: `2d30d957891025d1c091a4a2af5baea343564eddf6f043102283d501f9b80468`.

The checkpoint and complete step records are retained under the ignored local `runs/prune` artifact directory.
