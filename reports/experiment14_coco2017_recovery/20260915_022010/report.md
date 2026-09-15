# COCO2017 recovery diagnostic

- Preset: `standard_recovery`
- Epochs: `10`
- Parent/pruned lineage was verified by SHA256 before training.
- Selection includes the input checkpoint, so training cannot silently replace it with a worse best.pt.

| Model | Input mAP50-95 | Selected mAP50-95 | Change | Selected checkpoint |
|---|---:|---:|---:|---|
| parent | 0.4633 | 0.4633 | +0.0000 | source |
| pruned | 0.1723 | 0.4144 | +0.2421 | best |

This diagnostic decides whether recovery training is sound. It does not establish a pruning-method result.