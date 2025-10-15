from pathlib import Path


import os
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
import random
import torch
import argparse
import importlib

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    # LDMTextToImagePipeline
    StableDiffusionPipeline,
)

import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from torch import nn
from tqdm.auto import tqdm
# from tqdm.autonotebook import tqdm
from accelerate import Accelerator
from einops import rearrange, reduce
from sklearn.metrics import r2_score

from packaging import version
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate.logging import get_logger
from diffusers.utils.import_utils import is_xformers_available
from src.eval.eval_utils import GlobVideoDatasetWithLabel, ari, get_mask_cosine_distance, \
    hungarian_algorithm, clevrtex_label_reading, movi_label_reading
from accelerate.utils import ProjectConfiguration, set_seed

from src.models.backbone import UNetEncoder
from src.models.slot_attn import MultiHeadSTEVESA

from src.eval.oclf_metrics.visualization import Segmentation
from src.eval.eval_datasets.coco_eval import COCOEval, COCOEvalCollator
from src.eval.eval_datasets.coco_eval import build_coco_eval_dataset
from src.eval.parser import parse_args

args = parse_args()

   
if args.dataset == 'coco':
    train_dataset, val_dataset, collate_fn = build_coco_eval_dataset(args)



val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.val_batch_size,
    shuffle=False,
    num_workers=args.dataloader_num_workers,
    collate_fn=collate_fn,
    drop_last=True
)

set_seed(args.seed)
torch.backends.cudnn.deterministic = True
accelerator = Accelerator(mixed_precision=args.mixed_precision)
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16


def load_model(args):
    # max_num_obj = test_dataset.max_cat_id

    if os.path.exists(os.path.join(args.ckpt_path, "UNetEncoder")):
        pretrain_backbone = False
        backbone = UNetEncoder.from_pretrained(
            args.ckpt_path, subfolder="UNetEncoder".lower())
        backbone = backbone.to(device=accelerator.device, dtype=weight_dtype)
    else:
        pretrain_backbone = True
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        class DINOBackbone(torch.nn.Module):
            def __init__(self, dinov2):
                super().__init__()
                self.dinov2 = dinov2

            def forward(self, x):
                enc_out = self.dinov2.forward_features(x)
                return rearrange(
                    enc_out["x_norm_patchtokens"], 
                    "b (h w ) c -> b c h w",
                    h=int(np.sqrt(enc_out["x_norm_patchtokens"].shape[-2]))
                )
        backbone = DINOBackbone(dinov2).to(device=accelerator.device, dtype=weight_dtype)


    pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name
        )

    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    scheduler = scheduler_class.from_config(pipeline.scheduler.config)
    if os.path.exists(os.path.join(args.ckpt_path, "UNet2DConditionModel".lower())):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name, subfolder="vae")
        unet_model = UNet2DConditionModel.from_pretrained(
            args.ckpt_path, subfolder="UNet2DConditionModel".lower())
        pipeline = StableDiffusionPipeline(
                vae=vae,
                tokenizer=pipeline.tokenizer,
                text_encoder=pipeline.text_encoder,
                unet=unet_model,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=None,
            )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name, scheduler=scheduler
            )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = None if args.seed is None else torch.Generator(
            device=accelerator.device).manual_seed(args.seed)

    slot_attn = MultiHeadSTEVESA.from_pretrained(
        args.ckpt_path, subfolder="MultiHeadSTEVESA".lower())
    slot_attn = slot_attn.to(device=accelerator.device, dtype=weight_dtype)
    
    if args.enable_xformers_memory_efficient_attention and not pretrain_backbone:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            backbone.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    return slot_attn, backbone, generator, pipeline

def generate_images(args, backbone, slot_attn, pipeline, generator, val_dataloader, pretrain_backbone=True, cfg=None):
    seed = 42
    item_id = 0

    progress_bar = tqdm(
            range(0, len(val_dataloader)),
            initial=0,
            desc="Steps",
            position=0, leave=True
        )
    args.exp_name = args.ckpt_path.split('/')[2]
    if args.exp_name == 'finetune':
        args.exp_name = args.ckpt_path.split('/')[3]
    cur_img_dir = f'./glass/output/fid/{args.exp_name}_{args.dataset}/{cfg}/fake/'

    os.makedirs(cur_img_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
            instance_mask = batch['instance_mask'].to(device=accelerator.device, dtype=torch.int64)
            instance_mask[instance_mask>=1] = 1
            if pretrain_backbone:
                pixel_values_vit = batch["pixel_values_vit"].to(device=accelerator.device, dtype=weight_dtype)
                feat = backbone(pixel_values_vit)
            else:
                feat = backbone(pixel_values)
        
            # slots, attns = slot_attn(feat[:, None])  # for the time dimension
            slots, attn = slot_attn(feat[:, None], batch['num_objs'], use_mask=False)  # for the time dimension
            
            slots = slots[:, 0]
            generator = torch.Generator(
                device=accelerator.device).manual_seed(seed)
            images_gen = pipeline(
                prompt_embeds=slots,
                negative_prompt_embeds=None,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=250,
                generator=generator,
                guidance_scale=cfg,
                output_type="pt",
            ).images

            for idx in range(0, len(pixel_values)):
                F.to_pil_image(images_gen[idx]).save(f'{cur_img_dir}/{item_id}.png')
                item_id += 1
            if item_id > 500:
                break
            progress_bar.update(1)
            print(f'Generated {item_id} images')


cfgs = [1.3]
for cfg in cfgs:
    slot_attn, backbone, generator, pipeline = load_model(args)
    generate_images(args, backbone=backbone, slot_attn=slot_attn, pipeline=pipeline, generator=generator, val_dataloader=val_dataloader, cfg=cfg)