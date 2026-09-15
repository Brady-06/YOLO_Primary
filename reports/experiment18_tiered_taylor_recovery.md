# Experiment 18: conservative recovery for the 5% stage

- Training data: 114,191 train2017 images, excluding the calibration and tune subsets.
- Selection data: the fixed 2,048-image train-derived tune set; COCO val2017 was reserved for the final measurement.
- Recipe: 5 epochs, AdamW, learning rate `1e-4`, BatchNorm updates enabled, no augmentation.
- Tune mAP50-95: raw `0.514366`; recovery best `0.489790`; recovery last `0.489790`.
- Selection guard therefore retained the raw checkpoint.
- Selected full COCO val2017 mAP50-95: `0.418059`.

This recovery recipe did not improve the pruned model. The recovery checkpoints remain as evidence and are not the selected result.
