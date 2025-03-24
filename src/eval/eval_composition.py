from pathlib import Path
import os
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
import random
import torch
import argparse
import importlib
import json
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
from src.eval.eval_datasets.coco_eval import build_coco_eval_dataset
from src.eval.parser import parse_args
from scipy.optimize import linear_sum_assignment
from src.eval.oclf_metrics.visualization import Segmentation
from torchvision.utils import draw_segmentation_masks 

args = parse_args()

if args.dataset== 'coco':
    num_class = 80

# coco gen mask w feat freeze sa
def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

def miou_helper(pred_masks:torch.Tensor, target_masks:torch.Tensor) -> torch.Tensor:
    max_classes = int(max(pred_masks.max(), target_masks.max()).item() + 1)
    target_masks = F.one_hot(target_masks.to(torch.int64), num_classes=max_classes).float()
    pred_masks = pred_masks.argmax(dim=-3)  # [B, N, H, W] --> [B, H, W]
    pred_masks = F.one_hot(pred_masks).float()

    pred_masks = pred_masks.permute(0, 3, 1, 2)
    target_masks= target_masks.permute(0, 3, 1, 2)

    return mask_iou(pred_masks, target_masks)

def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor,) -> torch.Tensor:
    """
    Inputs:
    mask1: BxNxHxW torch.float32. Consists of [0, 1]
    mask2: BxMxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: BxNxM torch.float32. Consists of [0 - 1]
    """
    B, N, H, W = mask1.shape
    B, M, H, W = mask2.shape

    mask1 = mask1.view(B, N, H * W)
    mask2 = mask2.view(B, M, H * W)

    intersection = torch.matmul(mask1, mask2.swapaxes(1, 2))

    area1 = mask1.sum(dim=2).unsqueeze(1)
    area2 = mask2.sum(dim=2).unsqueeze(1)

    union = (area1.swapaxes(1, 2) + area2) - intersection


    iou = intersection / (union + 1e-8)  # [N, M]
    iou = 1 - iou.detach().cpu()

    return iou

def match(slots, predicted_masks, gt_masks, num_objs):
    iou_scores = miou_helper(predicted_masks, gt_masks)
    sizes = num_objs
    bs = iou_scores.shape[0]
    num_queries = slots.shape[1]
    pad_rows = slots.shape[1] - iou_scores.shape[1]
    if pad_rows > 0:
        iou_scores = torch.concat((iou_scores, torch.ones(bs, pad_rows, iou_scores.shape[-1]).to(iou_scores.device)), axis=1)
        
    C = iou_scores
    C = C.view(bs, num_queries, -1).cpu()
    indices = []
    # sizes indicate the numbers of slots we need to assign the true label to.
    for i, c in enumerate(C):
        indices.append(linear_sum_assignment(c[:sizes[i]]))
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] 
   
if args.dataset == 'coco':
    train_dataset, val_dataset, collate_fn = build_coco_eval_dataset(args, suppress_idx=True)
    train_anno_dir = os.path.join(args.dataset_root, 'annotations', f'instances_train2017.json')
    coco_train = COCO(train_anno_dir)
    val_anno_dir = os.path.join(args.dataset_root, 'annotations', f'instances_val2017.json')
    coco_val = COCO(val_anno_dir)
    # coco_lbls = [lbl-1 for lbl in coco_lbl]
    cat_ids = coco_val.getCatIds()
    cats = sorted(coco_val.loadCats(cat_ids), key=lambda x: x['id'])
    num_classes = len(cats)
    # cat_id is an original cat id,coco_label is set from 0 to 79
    cat_id_to_cat_name = {cat['id']: cat['name'] for cat in cats}
    coco_label_to_cat_id = {
        i: cat['id']
        for i, cat in enumerate(cats)
    }

    coco_label_to_cat_name = {
        coco_label: cat_id_to_cat_name[cat_id]
        for coco_label, cat_id in coco_label_to_cat_id.items()
    }



edit_type = 'add'
if edit_type in ['add', 'swap']:
    batch_size = 2
else:
    batch_size = 1
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
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

    # todo: this is ugly solution
    # use a more efficient scheduler at test time
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

def swap_slots(slots_x, slots_y, attn_x, attn_y, batch, vis_seg=None):
    attn_x = torch.nn.functional.interpolate(attn_x, size=256, mode='bilinear')
    attn_x = torch.nn.functional.one_hot(torch.argmax(attn_x, 1)).permute(-1, 0, 1, 2).to(torch.bool)
    attn_y = torch.nn.functional.interpolate(attn_y, size=256, mode='bilinear')
    attn_y = torch.nn.functional.one_hot(torch.argmax(attn_y, 1)).permute(-1, 0, 1, 2).to(torch.bool)

    x_idx= np.random.choice(attn_x.shape[0]-1)
    y_idx = np.random.choice(attn_y.shape[1])
    tmp_slot = slots_y[0][y_idx]
    slots_y[0][y_idx] = slots_x[0][x_idx]
    slots_x[0][x_idx] = tmp_slot
    return slots_x, attn_x[x_idx, 0, :, :], slots_y, attn_y[y_idx, 0, :, :]

def add_slots(slots_x, slots_y, attn_x, attn_y, batch, vis_seg=None):
    attn_x = torch.nn.functional.interpolate(attn_x, size=256, mode='bilinear')
    attn_x = torch.nn.functional.one_hot(torch.argmax(attn_x, 1)).permute(-1, 0, 1, 2).to(torch.bool)
    x_idx= np.random.choice(attn_x.shape[0]-1)
    y_idx = np.random.choice(slots_y.shape[1])
    slots_y[0][y_idx] = slots_x[0][x_idx]
    return slots_y, attn_x[x_idx, 0, :, :]

def remove_slot(slots, attn, batch, vis_seg=None):
    all_slots = []
    
    attn = torch.nn.functional.interpolate(attn, size=256, mode='bilinear')
    attn = torch.nn.functional.one_hot(torch.argmax(attn, 1)).permute(-1, 0, 1, 2).to(torch.bool)
    remove_idx = np.random.choice(attn.shape[0]-1)
    slots[0][remove_idx] = torch.zeros_like(slots[0][0])
    if len(slots) == 1:
        return slots.unsqueeze(0), attn[remove_idx, 0, :, :]
    else:
        return slots, attn[remove_idx, 0, :, :]

def remove_slot_seq(slots, attn, batch, vis_seg=None):
    all_slots = []
    
    for i in range(0, len(slots)):
        for idx in range(1, slots.shape[1]):
            slots[i][idx] = torch.zeros_like(slots[i][0])
            all_slots.append(slots[i])
    all_slots = torch.stack(all_slots).unsqueeze(1)
    return  all_slots 

def generate_images(args, backbone, slot_attn, pipeline, generator, val_dataloader, pretrain_backbone=True, sequential=False, edit_type=None):
    progress_bar = tqdm(
            range(0, len(val_dataloader)),
            initial=0,
            desc="Steps",
            position=0, leave=True
        )
    args.exp_name = args.ckpt_path.split('/')[2]
    if args.exp_name == 'finetune':
        args.exp_name = args.ckpt_path.split('/')[3]
    cur_img_dir = f'./image_generation/compositional/{edit_type}/{args.exp_name}/'
    os.makedirs(cur_img_dir, exist_ok=True)

    vis_seg = Segmentation()

    base_img_dir = f'./image_generation/compositional/{edit_type}/{args.exp_name}'
    # base_img_dir = '/visinf/home/ksingh/latent-slot-diffusion/original'
    os.makedirs(base_img_dir, exist_ok=True)
    cfg = 3
    num_inference_steps = 75
    seed = 42
    item_id = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
            if pretrain_backbone:
                pixel_values_vit = batch["pixel_values_vit"].to(device=accelerator.device, dtype=weight_dtype)
                feat = backbone(pixel_values_vit)
            else:
                feat = backbone(pixel_values)
           
            slots, attns = slot_attn(feat[:, None])  # for the time dimension
            slots = slots[:, 0]
            attns = attns[:, 0]
            labels = batch['annos'][:, :, 4]
            labels = [label-1 for label in labels[:]]
            all_label_names = []
            for label in labels:
                labels_names = []
                for lbl in label:
                    if lbl > 0:
                        labels_names.append(coco_label_to_cat_name[lbl.item()])
                all_label_names.append(labels_names)

            generator = torch.Generator(
                device=accelerator.device).manual_seed(seed)
            images_gen = pipeline(
                    # .to(dtype=torch.float32) # needed?
                    prompt_embeds=slots,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    guidance_scale=cfg,
                    output_type="pt",
                ).images
            attns = reduce(
                attns, 'b num_h (h w) s -> b s h w', h=int(np.sqrt(attns.shape[-2])), 
                reduction='mean'
            )
            if edit_type == 'add':
                slots_x = slots[0, ...].unsqueeze(0)
                attn_x =  attns[0, ...].unsqueeze(0)
                slots_y = slots[1, ...].unsqueeze(0)
                attn_y =  attns[1, ...].unsqueeze(0)

                slots, attns = add_slots(slots_x, slots_y, attn_x, attn_y, batch=batch, vis_seg=vis_seg)
                images_gen_edited = pipeline(
                        # .to(dtype=torch.float32) # needed?
                        prompt_embeds=slots,
                        height=args.resolution,
                        width=args.resolution,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        guidance_scale=cfg,
                        output_type="pt",
                    ).images
                
                original_img = pixel_values * 0.5 + 0.5

                # F.to_pil_image(images_gen[0]).save(f'{base_img_dir}/{item_id}/src_recon.png')

                os.makedirs(f'{cur_img_dir}/{item_id}', exist_ok=True)
                overlay = draw_segmentation_masks(original_img[0], attns, alpha=0.5, colors='red')
                F.to_pil_image(overlay).save(f'{cur_img_dir}/{item_id}/original_src.png')
                F.to_pil_image(images_gen[1]).save(f'{base_img_dir}/{item_id}/original_dst.png')
                F.to_pil_image(images_gen_edited[0]).save(f'{base_img_dir}/{item_id}/edited_dst.png')

            if edit_type == 'swap':
                slots_x = slots[0, ...].unsqueeze(0)
                attn_x =  attns[0, ...].unsqueeze(0)
                slots_y = slots[1, ...].unsqueeze(0)
                attn_y =  attns[1, ...].unsqueeze(0)

                slots_x, attn_x, slots_y, attn_y = swap_slots(slots_x, slots_y, attn_x, attn_y, batch=batch, vis_seg=vis_seg)
                images_gen_edited_x = pipeline(
                        # .to(dtype=torch.float32) # needed?
                        prompt_embeds=slots_x,
                        height=args.resolution,
                        width=args.resolution,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        guidance_scale=cfg,
                        output_type="pt",
                    ).images

                images_gen_edited_y = pipeline(
                        # .to(dtype=torch.float32) # needed?
                        prompt_embeds=slots_y,
                        height=args.resolution,
                        width=args.resolution,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        guidance_scale=cfg,
                        output_type="pt",
                    ).images

                
                original_img = pixel_values * 0.5 + 0.5

                # F.to_pil_image(images_gen[0]).save(f'{base_img_dir}/{item_id}/src_recon.png')

                os.makedirs(f'{cur_img_dir}/{item_id}', exist_ok=True)
                overlay_x = draw_segmentation_masks(original_img[0], attn_x, alpha=0.5, colors='red')
                overlay_y = draw_segmentation_masks(original_img[1], attn_y, alpha=0.5, colors='blue')

                F.to_pil_image(overlay_x).save(f'{cur_img_dir}/{item_id}/original_src_x.png')
                F.to_pil_image(overlay_y).save(f'{cur_img_dir}/{item_id}/original_src_y.png')

                F.to_pil_image(images_gen_edited_x[0]).save(f'{base_img_dir}/{item_id}/edited_x.png')
                F.to_pil_image(images_gen_edited_y[0]).save(f'{base_img_dir}/{item_id}/edited_y.png')

            sequential = False
            if edit_type == 'remove':
                if sequential == False:
                    original_attn = attns
                    slots, attns = remove_slot(slots, attns, batch=batch, vis_seg=vis_seg)
                    slots = slots[:, 0]
                    images_gen_edited = pipeline(
                        # .to(dtype=torch.float33) # needed?
                        prompt_embeds=slots,
                        height=args.resolution,
                        width=args.resolution,
                        num_inference_steps=20,
                        generator=generator,
                        guidance_scale=cfg,
                        output_type="pt",
                    ).images


                os.makedirs(f'{cur_img_dir}/{item_id}', exist_ok=True)
                for idx in range(0, len(images_gen_edited)):
                    F.to_pil_image(images_gen_edited[idx]).save(f'{cur_img_dir}/{item_id}/edited_{idx}.png')
                    original_img = pixel_values * 0.5 + 0.5
                    overlay_target_only = draw_segmentation_masks(original_img[0], attns, alpha=0.5, colors='red')
                    original_attn = torch.nn.functional.interpolate(original_attn, size=256, mode='bilinear').to('cpu')
                    overlay_all = torchvision.transforms.functional.to_pil_image(vis_seg(original_img, original_attn).n_instances)
                    
                    F.to_pil_image(overlay_target_only).save(f'{cur_img_dir}/{item_id}/slot_removed_{idx}.png')
                    overlay_all.save(f'{cur_img_dir}/{item_id}/sm_{idx}.png')

                F.to_pil_image(images_gen[0]).save(f'{base_img_dir}/{item_id}/original.png')
            item_id += 1
            if item_id > 100:
                break
            progress_bar.update(1)
            
slot_attn, backbone, generator, pipeline = load_model(args)
generate_images(args, backbone=backbone, slot_attn=slot_attn, pipeline=pipeline, generator=generator, val_dataloader=val_dataloader, sequential=False, edit_type=edit_type)