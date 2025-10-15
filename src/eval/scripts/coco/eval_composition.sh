#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0  python src/eval/eval_composition.py \
--ckpt_path "./glass/glass_st2/glass/checkpoint" \
--output_dir ./outputs \
--enable_xformers_memory_efficient_attention --num_workers 4  --dataset coco --dataset_root /fastdata/ksingh/coco2017  --val_batch_size 1
