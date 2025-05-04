import argparse
import logging
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

logger = get_logger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data_root, tokenizer, size=512, center_crop=False, random_flip=False, caption_column="text"):
        self.data_root = Path(data_root)
        self.image_paths = list(self.data_root.glob("*.jpg")) + list(self.data_root.glob("*.png"))
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.caption_column = caption_column

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transforms(image)

        text_path = image_path.with_suffix(".txt")
        if text_path.exists():
            with open(text_path, "r") as f:
                text = f.read().strip()
        else:
            text = "A photo"

        inputs = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": image, "input_ids": inputs}

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for Text-to-Image")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    unet = get_peft_model(unet, lora_config)

    dataset = CustomDataset(
        args.train_data_dir, tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        caption_column=args.caption_column
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=args.dataloader_num_workers
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    for epoch in range(100000):  # effectively infinite loop
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                outputs = unet(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"])
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % args.checkpointing_steps == 0:
                accelerator.save_state(args.output_dir)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.save_state(args.output_dir)

    if args.validation_prompt:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer
        )
        pipe.to(accelerator.device)
        image = pipe(args.validation_prompt).images[0]
        image.save(os.path.join(args.output_dir, "validation.png"))

if __name__ == "__main__":
    main()
