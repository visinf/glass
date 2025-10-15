#!/usr/bin/bash
declare -a chkpt_paths=(
   "./glass/glass_st2/glass/checkpoint" \
)
for i in "${chkpt_paths[@]}" 
do
  echo "Evaluating $i";
  python src/eval/eval_generation.py --ckpt_path $i --seed 42 --val_batch_size 64 --dataset coco --dataset_root /fastdata/ksingh/coco2017 --output_dir ./outputs
done