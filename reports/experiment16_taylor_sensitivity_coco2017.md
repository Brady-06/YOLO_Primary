# Experiment 16: COCO2017 Taylor sensitivity

- Source: official YOLO11s, SHA256 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.
- Split: 2,048 calibration images and a disjoint 2,048-image tune set from train2017.
- Probe: remove 8 lowest-Taylor channels from each structurally safe candidate, always starting from the same source checkpoint.
- Result: 42 candidates tested; 35 feasible and 7 rejected by dependency protection.
- Ranking metric: tune-set mAP50-95 loss per actual GMAC saved.

The full ranking and split hashes are retained under the ignored local `runs/analysis` artifact directory.
