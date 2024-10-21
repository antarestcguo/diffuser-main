from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import random
from torchvision import transforms
from pathlib import Path
import numpy as np
import os
import torch


class TCControlDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts and control images
    """

    def __init__(
            self,
            instance_dir,
            control_dir,
            instance_file_name,
            caption_flag_str,
            placeholder_token,
            tokenizer,
            size=512,
    ):
        self.caption_flag_str = caption_flag_str
        self.placeholder_token = placeholder_token  # just for one_caption augmentation
        self.instance_name_list = []
        self.instance_caption_list = []
        self.control_name_list = []
        self.other_name_list = []
        self.other_caption_list = []
        with open(instance_file_name, 'r') as f:
            for line in f.readlines():  # image_name, caption, control_name
                tokens = line.strip().split('\t')
                self.instance_name_list.append(os.path.join(instance_dir, tokens[0]))
                self.instance_caption_list.append(tokens[1].replace('unique_identifier', placeholder_token))
                self.control_name_list.append(os.path.join(control_dir, tokens[2]))

        # aug_text, ori_caption,
        self.size = size
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR

        self.num_instance_images = len(self.instance_name_list)
        self.num_other_images = len(self.other_name_list)
        self._length = max(self.num_other_images, self.num_instance_images)

        self.image_transforms = transforms.Compose(
            [
                # self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.control_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        if self.caption_flag_str == 'one_caption':
            placeholder_string = self.placeholder_token
            text = random.choice(self.imagenet_templates_small).format(placeholder_string)
        elif self.caption_flag_str == 'ori_caption':
            text = self.instance_caption_list[index]

        example["instance_prompt_ids"] = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids  # custom input_ids, textual inversion input_ids[0]

        instance_image = self.instance_name_list[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        control_image = self.control_name_list[index % self.num_instance_images]
        control_image = Image.open(control_image)
        if not control_image.mode == "RGB":
            control_image = control_image.convert("RGB")

        instance_image = self.image_transforms(instance_image)
        control_image = self.image_transforms(control_image)

        example["instance_images"] = instance_image
        example["control_images"] = control_image  # not normalize

        return example
