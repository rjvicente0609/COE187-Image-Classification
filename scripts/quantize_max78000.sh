#!/bin/sh
# Generated Log folder during the Training for Flower Dataset
# Update the LOG_DIRECTORY with your actual flower dataset log folder name
LOG_DIRECTORY="../ai8x-training/logs/2025.11.06-200027"

# Quantization for Flower Dataset
python quantize.py $LOG_DIRECTORY/qat_best.pth.tar $LOG_DIRECTORY/qat_best-quantized.pth.tar --device MAX78000 -v "$@"

# Quantization for cats and dogs
#LOG_DIRECTORY="C:\Users\Siote\Documents\COE187\ai8x-training\logs\2025.11.06-200027"
#python quantize.py $LOG_DIRECTORY/qat_best.pth.tar $LOG_DIRECTORY/qat_best-quantized.pth.tar --device MAX78000 -v "$@"

# Quantization for Key Word Spotted
#python quantize.py ../ai8x-training/logs/2023.04.06-172201_kws/qat_best.pth.tar ../ai8x-training/logs/2023.04.06-172201_kws/qat_best-q.pth.tar --device MAX78000 -v "$@"

# Quantization for Emotion Recognition 

# Common Template for Quantize Scripts
# python quantize.py ../ai8x-training/logs/<log folder name of your trained model>/qat_best.pth.tar ../ai8x-training/logs/<log folder name of your trained model>/qat_best-quantized.pth.tar --device MAX78000 -v "$@"

