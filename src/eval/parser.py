import argparse

def parse_args(input_args=None):
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
        "--dataset",
        type=str,
        default='coco',
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
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
        "--path_to_coco",
        type=str,
        default="path_to_coco",
        help="Path to coco dataset.",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Path to the dataset root.",
        required=True,
    )

    parser.add_argument(
        "--val_batch_size", type=int, default=4, 
        help="Batch size (per device) for the validation dataloader."
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--joint_training",
        action="store_true",
        help=(
            "Jointly trained diffusion decoder and sa modules"
        )
    )



    parser.add_argument(
        "--use_boxes",
        action="store_true",
    )

    parser.add_argument(
        "--foreground_only",
        action="store_true"
    )

    args = parser.parse_args()

    return args