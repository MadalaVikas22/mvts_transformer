#!/bin/bash

#mkdir -p experiments

python src/main.py \
  --output_dir experiments \
  --comment "OCO Cognitive Load Classification" \
  --name OCO_ThreeClass_MVTS \
  --records_file OCO_Classification_records.xls \
  --data_dir /Users/vikasvicky/Documents/Counterfactuals_for_TimeSeries/cogload/oco_small \
  --data_class cogload \
  --pattern "*" \
  --val_ratio 0.2 \
  --test_ratio 0.1 \
  --epochs 1 \
  --lr 0.001 \
  --optimizer RAdam \
  --pos_encoding learnable \
  --task classification \
  --key_metric accuracy \
  --label_column target_threeclass \
  --batch_size 8 \
  --num_workers 0