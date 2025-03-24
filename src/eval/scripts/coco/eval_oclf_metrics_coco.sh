#!/usr/bin/bash
declare -a chkpt_paths=(
  "./glass/glass_st1/glass/checkpoint"
)
for i in "${chkpt_paths[@]}" 
do
  python src/eval/oclf_metrics/eval_metrics.py --ckpt_path $i --seed 42 --batch_size 64 --dataset coco --dataset_root /fastdata/ksingh/coco2017 --backbone_config pretrain_dino
done