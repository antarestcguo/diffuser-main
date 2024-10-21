from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import random
from torchvision import transforms
from pathlib import Path
import numpy as np
import os
import torch
import cv2


# for random mask
def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))  # h,w,c
    image = image[None].transpose(0, 3, 1, 2)  # 1,c,h,w
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))  # h,w
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]  # 1,1,h,w
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)  # 1,1,h,w

    masked_image = image * (mask < 0.5)  # 1,c,h,2

    return mask, masked_image


# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask


class TCCustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    imagenet_templates_small = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    def __init__(
            self,
            instance_dir,
            other_dir,
            instance_file_name,
            other_file_name,
            caption_flag_str,
            placeholder_token,
            tokenizer,
            size=512,
            mask_size=64,
            center_crop=False,
            num_class_images=200,
            hflip=False,
            aug=True,
            b_other=False,
    ):
        self.caption_flag_str = caption_flag_str
        self.placeholder_token = placeholder_token  # just for one_caption augmentation
        self.b_other = b_other
        self.instance_name_list = []
        self.instance_caption_list = []
        self.other_name_list = []
        self.other_caption_list = []
        with open(instance_file_name, 'r') as f:
            for line in f.readlines():
                tokens = line.strip().split('\t')
                self.instance_name_list.append(os.path.join(instance_dir, tokens[0]))
                self.instance_caption_list.append(tokens[1].replace('unique_identifier', placeholder_token))

        if self.b_other:
            with open(other_file_name, 'r') as f:
                for line in f.readlines():
                    tokens = line.strip().split('\t')
                    self.other_name_list.append(os.path.join(other_dir, tokens[0]))
                    self.other_caption_list.append(tokens[1])
                    if len(self.other_name_list) > num_class_images:
                        break

        # aug_text, ori_caption,
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.num_instance_images = len(self.instance_name_list)
        self.num_other_images = len(self.other_name_list)
        self._length = max(self.num_other_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top: top + inner, left: left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top: top + inner, left: left + inner, :] = image
            mask[
            top // factor + 1: (top + scale) // factor - 1, left // factor + 1: (left + scale) // factor - 1
            ] = 1.0
        return instance_image, mask

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
        instance_image = self.flip(instance_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)

        if self.b_other:
            class_image = self.other_name_list[index % self.num_class_images]
            class_prompt = self.other_caption_list[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


class TCCustomDiffusionCleanDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    imagenet_templates_small = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        # "a photo of a dirty {}",
        # "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    def __init__(
            self,
            instance_dir,
            instance_file_name,
            caption_flag_str,
            placeholder_token,
            tokenizer,
            # encoder,
            size=512,
            mask_size=64,
            center_crop=False,
            hflip=False,
            aug=True,
    ):
        self.caption_flag_str = caption_flag_str
        self.placeholder_token = placeholder_token  # just for one_caption augmentation
        self.instance_name_list = []
        self.instance_caption_list = []

        with open(instance_file_name, 'r') as f:
            for line in f.readlines():
                tokens = line.strip().split('\t')
                self.instance_name_list.append(os.path.join(instance_dir, tokens[0]))
                self.instance_caption_list.append(tokens[1].replace('unique_identifier', placeholder_token))

        # aug_text, ori_caption,
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        # self.encoder = encoder
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.num_instance_images = len(self.instance_name_list)
        self._length = self.num_instance_images
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top: top + inner, left: left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top: top + inner, left: left + inner, :] = image
            mask[
            top // factor + 1: (top + scale) // factor - 1, left // factor + 1: (left + scale) // factor - 1
            ] = 1.0
        return instance_image, mask

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
        instance_image = self.flip(instance_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)

        return example


class TCCustomDiffusionCleanMaskDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    imagenet_templates_small = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        # "a photo of a dirty {}",
        # "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    def __init__(
            self,
            instance_dir,
            mask_dir,
            instance_file_name,
            caption_flag_str,
            placeholder_token,
            tokenizer,
            # encoder,
            size=512,
            mask_size=64,
            center_crop=False,
            hflip=False,  # useless
            aug=True,
    ):
        self.caption_flag_str = caption_flag_str
        self.placeholder_token = placeholder_token  # just for one_caption augmentation
        self.instance_name_list = []
        self.instance_caption_list = []
        self.mask_name_list = []

        with open(instance_file_name, 'r') as f:
            for line in f.readlines():
                tokens = line.strip().split('\t')
                self.instance_name_list.append(os.path.join(instance_dir, tokens[0]))
                self.instance_caption_list.append(tokens[1].replace('unique_identifier', placeholder_token))
                self.mask_name_list.append(os.path.join(mask_dir, tokens[2]))

        # aug_text, ori_caption,
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        # self.encoder = encoder
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.num_instance_images = len(self.instance_name_list)
        self._length = self.num_instance_images
        self.flip = transforms.RandomHorizontalFlip(1.0)

    def __len__(self):
        return self._length

    def preprocess(self, image, mask_image, scale, resample):
        # 1. find max side
        w, h = image.size
        if h > w:
            max_side = h
        else:
            max_side = w
        # 2. compute resize_h and resize_w
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size

        resize_h = int(scale * h / float(max_side))
        resize_w = int(scale * w / float(max_side))

        # 3. resize original image and mask,then prepare the instance_image and instance_mask
        image = image.resize((resize_w, resize_h), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        mask_image = mask_image.resize((resize_w, resize_h), resample=resample)
        mask_image = np.array(mask_image).astype(np.float32) / 255.0  # shape:h,w,3

        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        instance_mask = np.zeros((self.size, self.size), dtype=np.float32)

        # 4. random top lef as random crop
        # mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            min_side = min(resize_h, resize_w)
            if min_side > self.size:
                top = np.random.randint(0, resize_h - self.size + 1)
                left = np.random.randint(0, resize_w - self.size + 1)
                instance_image = image[top: top + self.size, left: left + self.size, :]
                instance_mask = mask_image[top: top + self.size, left: left + self.size,
                                0]  # shape:h,w,3,only need one channel
            else:
                if min_side == resize_h:
                    top = np.random.randint(0, self.size - resize_h + 1)
                    left = np.random.randint(0, resize_w - self.size + 1)

                    instance_image[top:top + resize_h, :, :] = image[:, left:left + self.size, :]
                    instance_mask[top:top + resize_h, :] = mask_image[:, left:left + self.size, 0]
                else:
                    top = np.random.randint(0, resize_h - self.size + 1)
                    left = np.random.randint(0, self.size - resize_w + 1)
                    instance_image[:, left:left + resize_w, :] = image[top:top + self.size, :, :]
                    instance_mask[:, left:left + resize_w] = mask_image[top:top + self.size, :, 0]

            # instance_image = image[top: top + inner, left: left + inner, :]
            # instance_mask = mask_image[top: top + inner, left: left + inner, 0]  # shape:h,w,3,only need one channel
            # mask = np.ones((self.size // factor, self.size // factor))
        else:
            top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
            instance_image[top: top + resize_h, left: left + resize_w, :] = image
            # mask[
            # top // factor + 1: (top + scale) // factor - 1, left // factor + 1: (left + scale) // factor - 1
            # ] = 1.0
            instance_mask[top: top + resize_h, left: left + resize_w] = mask_image[:, :,
                                                                        0]  # shape:h,w,3,only need one channel

        # 5. resize instance_mask to the mask size
        mask = cv2.resize(instance_mask, (self.size // factor, self.size // factor))
        mask = np.ones_like(mask) * (mask > 0.5)
        return instance_image, mask

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
        mask_image_name = self.mask_name_list[index % self.num_instance_images]
        mask_image = Image.open(mask_image_name)

        if np.random.uniform() > 0.5:
            instance_image = self.flip(instance_image)
            mask_image = self.flip(mask_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                # else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
                else np.random.randint(int(0.8 * self.size), int(1.2 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, mask_image, random_scale, self.interpolation)

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)

        return example

    def debug_idx(self, index):
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
        mask_image_name = self.mask_name_list[index % self.num_instance_images]
        mask_image = Image.open(mask_image_name)

        if np.random.uniform() > 0.5:
            instance_image = self.flip(instance_image)
            mask_image = self.flip(mask_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, mask_image, random_scale, self.interpolation)

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)

        return example


class TCCustomDiffusionRMaskInpaintingDataset(Dataset):  # random mask
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    imagenet_templates_small = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    def __init__(
            self,
            instance_dir,
            other_dir,
            instance_file_name,
            other_file_name,
            caption_flag_str,
            placeholder_token,
            tokenizer,
            size=512,
            mask_size=64,
            center_crop=False,
            num_class_images=200,
            hflip=False,
            aug=True,
            b_other=False,
    ):
        self.caption_flag_str = caption_flag_str
        self.placeholder_token = placeholder_token  # just for one_caption augmentation
        self.b_other = b_other
        self.instance_name_list = []
        self.instance_caption_list = []
        self.other_name_list = []
        self.other_caption_list = []
        with open(instance_file_name, 'r') as f:
            for line in f.readlines():
                tokens = line.strip().split('\t')
                self.instance_name_list.append(os.path.join(instance_dir, tokens[0]))
                self.instance_caption_list.append(tokens[1].replace('unique_identifier', placeholder_token))

        if self.b_other:
            with open(other_file_name, 'r') as f:
                for line in f.readlines():
                    tokens = line.strip().split('\t')
                    self.other_name_list.append(os.path.join(other_dir, tokens[0]))
                    self.other_caption_list.append(tokens[1])
                    if len(self.other_name_list) > num_class_images:
                        break

        # aug_text, ori_caption,
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.num_instance_images = len(self.instance_name_list)
        self.num_other_images = len(self.other_name_list)
        self._length = max(self.num_other_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
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
        instance_image = self.image_transforms_resize_and_crop(instance_image)

        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)

        if self.b_other:
            class_image = self.other_name_list[index % self.num_class_images]
            class_prompt = self.other_caption_list[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


class TCCustomDiffusionFixedMaskInpaintingDataset(Dataset):  # random mask
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    imagenet_templates_small = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    def __init__(
            self,
            instance_dir,
            other_dir,
            mask_dir,
            instance_file_name,
            other_file_name,
            caption_flag_str,
            placeholder_token,
            tokenizer,
            size=512,
            mask_size=64,  # useless
            center_crop=False,
            num_class_images=200,
            hflip=False,
            aug=True,
            b_other=False,
    ):
        self.caption_flag_str = caption_flag_str
        self.placeholder_token = placeholder_token  # just for one_caption augmentation
        self.b_other = b_other
        self.instance_name_list = []
        self.instance_caption_list = []
        self.mask_name_list = []
        self.other_name_list = []
        self.other_caption_list = []
        with open(instance_file_name, 'r') as f:
            for line in f.readlines():
                tokens = line.strip().split('\t')
                self.instance_name_list.append(os.path.join(instance_dir, tokens[0]))
                self.instance_caption_list.append(tokens[1].replace('unique_identifier', placeholder_token))
                self.mask_name_list.append(os.path.join(mask_dir, tokens[2]))

        if self.b_other:
            with open(other_file_name, 'r') as f:
                for line in f.readlines():
                    tokens = line.strip().split('\t')
                    self.other_name_list.append(os.path.join(other_dir, tokens[0]))
                    self.other_caption_list.append(tokens[1])
                    if len(self.other_name_list) > num_class_images:
                        break

        # aug_text, ori_caption,
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.num_instance_images = len(self.instance_name_list)
        self.num_other_images = len(self.other_name_list)
        self._length = max(self.num_other_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
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
        instance_image = self.image_transforms_resize_and_crop(instance_image)

        instance_image = self.image_transforms(instance_image)  # c,h,w

        mask_image_name = self.mask_name_list[index % self.num_instance_images]
        mask_image = Image.open(mask_image_name)
        if not mask_image.mode == "RGB":
            mask_image = mask_image.convert("RGB")
        mask_image = self.image_transforms_resize_and_crop(mask_image)

        # covert mask to tensor
        mask = np.array(mask_image.convert("L"))  # h,w
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]  # 1,1,h,w
        mask = torch.from_numpy(mask)
        masked_image = instance_image[None] * (mask < 0.5)  # 1,c,h,w

        example["mask"] = mask
        example["instance_images"] = instance_image
        example['masked_image'] = masked_image
        # mask, instance_image, masked_image are the tensor, with the shape (c,H,W)

        if self.b_other:
            class_image = self.other_name_list[index % self.num_class_images]
            class_prompt = self.other_caption_list[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example
