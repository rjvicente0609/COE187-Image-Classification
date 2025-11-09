#!/bin/sh
MODEL="ai85flowersnet"
DATASET="flowers"
QUANTIZED_MODEL="logs/2025.11.06-200027/qat_best-quantized.pth.tar"

# evaluate scripts for flowers
python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL  -8 --save-sample 1 --device MAX78000 "$@"

#evaluate scripts for kws
# python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL -8 --save-sample 1 --device MAX78000 "$@"

