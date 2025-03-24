import argparse
import sys
sys.path.append('./')
import importlib
import numpy as np
import torch
from packaging import version

from accelerate import Accelerator
from einops import rearrange, reduce
import diffusers
from diffusers import StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
import os
import torch.nn.functional as F
import torchvision
from PIL import Image

from src.eval.eval_datasets.coco_eval import build_coco_eval_dataset
from src.models.backbone import UNetEncoder
from src.models.slot_attn import MultiHeadSTEVESA
from src.eval.oclf_metrics.oclf_eval_masks import *
from src.eval.oclf_metrics.visualization import Segmentation

import random
import os
import json
import tqdm

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_all_seeds(42)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a checkpoint folder for the model.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=666,
    )

    parser.add_argument(
        "--backbone_config",
        type=str,
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--validation_scheduler",
        type=str,
        default="DPMSolverMultistepScheduler",
        choices=["DPMSolverMultistepScheduler", "DDPMScheduler"],
        help="Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.",
    )

    parser.add_argument(
        "--vit_input_resolution",
        type=int,
        default=448,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs/images",
        help="Path to a output folder for logging images.",
    )

    parser.add_argument(
        "--cfg_list",
        nargs='+', type=str, metavar=('x', 'y'),
        default=(1.0, 1.1, 1.3, 1.5, 2.0, 3.0),
        help="List of classifier free guidance values.",
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help=(
            "Whether to use the memory efficient attention implementation of xFormers. This is an experimental feature"
            " and is only available for PyTorch >= 1.10.0 and xFormers >= 0.0.17."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default='coco'
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/fastdata/ksingh/VOC",
        help="Path to the dataset root.",
        required=True
    )

    parser.add_argument(
        "--use_boxes",
        action="store_true",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=False
    )

    parser.add_argument(
        "--dynamic",
        action="store_true"
    )

    parser.add_argument(
        "--init_mlp",
        action="store_true"
    )

    parser.add_argument(
        "--return_image",
        action="store_true"
    )
    parser.add_argument(
        "--ignore_background",
        action="store_true"
    )



    args = parser.parse_args()
    return args


args = parse_args()


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
    elif args.backbone_config == "pretrain_dino":
        train_backbone = False
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
        backbone = DINOBackbone(dinov2)
    else:
        raise ValueError(
            f"Unknown unet config {args.unet_config}")

    pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name
        )

    # todo: this is ugly solution
    # use a more efficient scheduler at test time
    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    scheduler = scheduler_class.from_config(pipeline.scheduler.config)

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
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            backbone.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")
 

    return slot_attn, backbone, generator, pipeline

def forward_pass(batch, backbone, slot_attn, pipeline, backbone_config, return_image=False, init_mlp=False, dynamic=False):
    with torch.no_grad():
        # image, true_mask, labels = batch_data
        pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
        if backbone_config == "pretrain_dino":
            pixel_values_vit = batch["pixel_values_vit"].to(device=accelerator.device, dtype=weight_dtype)
            feat = backbone(pixel_values_vit)
        else:
            feat = backbone(pixel_values)

        if not init_mlp:
            slots, attns = slot_attn(feat[:, None], batch['num_objs'], use_mask=dynamic)  # for the time dimension
        else:
            annos=batch['annos'].numpy()[..., :4]
            max_annos_num = max(anno.shape[0] for anno in annos)
            max_annos_num = max(7, max_annos_num) # should be atleast num_slots
            if max_annos_num > 0:
                input_annos = np.ones(
                    (len(annos), max_annos_num, 4), dtype=np.float32) * (-1)
                for i, anno in enumerate(annos):
                    if anno.shape[0] > 0:
                        input_annos[i, :anno.shape[0], :] = anno
            else:
                input_annos = np.ones((len(annos), 1, 4), dtype=np.float32) * (-1)
            input_annos = torch.from_numpy(input_annos).float().to(feat.device)
            num_objs = batch['num_objs']
            slots, attns = slot_attn(feat[:, None], num_objs, use_mask=dynamic, bbox=input_annos)  # for the time dimension

        slots = slots[:, 0]
        attns = attns[:, 0]
        attns = reduce(
            attns, 'b num_h (h w) s -> b s h w', h=int(np.sqrt(attns.shape[-2])), 
            reduction='mean'
        )

        return slots, attns


def run_metric(args):
    if args.dataset == 'coco':
        train_dataset, val_dataset, collate_fn = build_coco_eval_dataset(args, suppress_idx=True)

    vis_seg = Segmentation()

    val_dataloader= torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
        drop_last=True
    )

    slot_attn, backbone, generator, pipeline = load_model(args)
    backbone.to(accelerator.device, dtype=weight_dtype)
    args.exp_name = args.ckpt_path.split('/')[2]
    metric_path = f'./glass/{args.exp_name}/output/'

    os.makedirs(metric_path, exist_ok=True)
    
    fg_ari_metric = ARIMetric(foreground=True, convert_target_one_hot=False, ignore_overlaps=True)
    ari_metric = ARIMetric(foreground=False, convert_target_one_hot=False, ignore_overlaps=True)
    instance_mask_iou = UnsupervisedMaskIoUMetric(use_threshold=False, ignore_overlaps=True, ignore_background=args.ignore_background)
    semantic_mask_iou = UnsupervisedMaskIoUMetric(use_threshold=False, ignore_overlaps=True, ignore_background=args.ignore_background)
    mbo_i = UnsupervisedMaskIoUMetric(use_threshold=False, ignore_overlaps=True, matching='best_overlap', ignore_background=args.ignore_background)
    mbo_c = UnsupervisedMaskIoUMetric(use_threshold=False, ignore_overlaps=True, matching='best_overlap', ignore_background=args.ignore_background)
    object_recovery = UnsupervisedMaskIoUMetric(use_threshold=False, ignore_overlaps=True, matching='best_overlap', compute_discovery_fraction=True, ignore_background=args.ignore_background)
    cor_loc =MaskCorLocMetric(use_threshold=False, ignore_overlaps=True,) 

    item_id = 0
    folder_id = 0
    pbar = tqdm.tqdm(total=len(val_dataloader))
    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
            pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
            slots, attn  = forward_pass(batch, backbone=backbone, slot_attn=slot_attn, pipeline=pipeline, backbone_config=args.backbone_config, init_mlp=args.init_mlp, dynamic=args.dynamic, return_image=args.return_image)
            instance_mask =  F.one_hot(batch['instance_mask'].to(torch.int64)).permute(0, 3, 1, 2)
            segmentation_mask =  F.one_hot(batch['segmentation_mask'].to(torch.int64)).permute(0, 3, 1, 2)
            attn = F.interpolate(attn, size=instance_mask.shape[-1], mode='bilinear').to('cpu')
            fg_ari_metric.update(attn, instance_mask)
            ari_metric.update(attn, segmentation_mask)
            instance_mask_iou.update(attn, instance_mask)
            semantic_mask_iou.update(attn, segmentation_mask)
            mbo_i.update(attn, instance_mask)
            mbo_c.update(attn, segmentation_mask)
            object_recovery.update(attn, instance_mask)
            cor_loc.update(attn, instance_mask)
            if args.return_image : 
                out_path = f'./glass/output/{args.exp_name}/{folder_id}'
                os.makedirs(out_path, exist_ok=True)
                if item_id % 100 == 0:
                    folder_id += 1
                im = torchvision.transforms.functional.to_pil_image(vis_seg(pixel_values * 0.5 + 0.5, attn).n_instances)
                im.save(os.path.join(out_path, f'{item_id}_segmentation.png'))
                item_id+=1
            pbar.update(1)
            

    out_dict = {'exp_name': args.exp_name, 'ckpt_path': args.ckpt_path, 'metrics': 
                {
                    'fg_ari': fg_ari_metric.compute().item(),
                    'ari': ari_metric.compute().item(),
                    'fg_iou': instance_mask_iou.compute().item(),
                    'iou': semantic_mask_iou.compute().item(),
                    'mbo_i': mbo_i.compute().item(),
                    'mbo_c': mbo_c.compute().item(),
                    'object_recovery': object_recovery.compute().item(),
                    'cor_loc': cor_loc.compute().item()
                }}
    with open(os.path.join(metric_path, f'metrics_{args.dataset}.json'), 'w') as f:
        json.dump(out_dict, f)
run_metric(args)