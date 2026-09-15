# Experiment 15: Taylor structured pruning on COCO2017

The parent is the official YOLO11s checkpoint recorded by SHA256.
Pruning actions were selected on a fixed training subset without using val2017.

| Metric | Parent | Taylor-pruned raw |
|---|---:|---:|
| Parameters | 9,458,752 | 8,516,768 |
| GMACs | 10.7991 | 9.7159 |
| GMAC reduction | 0.00% | 10.03% |
| mAP50-95 | 0.4633 | 0.1723 |

Stop reason: `target_reached`.
Raw mAP50-95 change: -0.2910.

Recovery training is deliberately separate. Run the recovery diagnostic with this run_info.json as provenance before choosing a deployable model.