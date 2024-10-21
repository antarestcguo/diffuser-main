import torch
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def compute_ca_loss(attn_maps_mid, attn_maps_up, bboxes, object_positions, height, width):
    loss = 0
    object_number = len(bboxes)
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated.chunk(2)[1]
        #
        b, i, j = attn_map.shape
        # H = W = int(math.sqrt(i))
        H = int(np.ceil(float(height) / 64))  # for mid level
        W = int(np.ceil(float(width) / 64))  # for mid level
        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            for obj_box in bboxes[obj_idx]:
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(
                    dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss / len(object_positions[obj_idx]))

    for attn_map_integrated in attn_maps_up[0]:
        attn_map = attn_map_integrated.chunk(2)[1]
        #
        b, i, j = attn_map.shape
        # H = W = int(math.sqrt(i))
        # for up level
        H = int(np.ceil(float(height) / 32))
        W = int(np.ceil(float(width) / 32))

        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            for obj_box in bboxes[obj_idx]:
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                # ca_map_obj = attn_map[:, :, object_positions[obj_position]].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(
                    dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss / len(object_positions[obj_idx]))
    loss = loss / (object_number * (len(attn_maps_up[0]) + len(attn_maps_mid)))
    return loss


def Pharse2idx(prompt, phrases):
    phrases = [x.strip() for x in phrases.split(';')]
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    for obj in phrases:
        obj_position = []
        for word in obj.split(' '):
            obj_first_index = prompt_list.index(word) + 1
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)

    return object_positions


def draw_box(pil_img, bboxes, phrases, save_path, img_w, img_h):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype('/home/Antares.Guo/code/diffusers/examples/layout/YeZiGongChangAoYeHei-2.ttf', 25)
    phrases = [x.strip() for x in phrases.split(';')]
    for obj_bboxes, phrase in zip(bboxes, phrases):
        for obj_bbox in obj_bboxes:
            x_0, y_0, x_1, y_1 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
            draw.rectangle([int(x_0 * img_w), int(y_0 * img_h), int(x_1 * img_w), int(y_1 * img_h)], outline='red',
                           width=5)
            draw.text((int(x_0 * img_w) + 5, int(y_0 * img_h) + 5), phrase, font=font, fill=(255, 0, 0))
    pil_img.save(save_path)


def setup_logger(save_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(os.path.join(save_path, f"{logger_name}.log"))
    file_handler.setLevel(logging.INFO)

    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


# RnB define
class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
            self,
            channels: int = 1,
            kernel_size: int = 3,
            sigma: float = 0.5,
            dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


smth_3 = GaussianSmoothing(sigma=3.0).cuda()

sobel_x = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float16).cuda()  # modify from float32 to float16

sobel_y = torch.tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=torch.float16).cuda()  # modify from float32 to float16

sobel_x = sobel_x.view(1, 1, 3, 3)
sobel_y = sobel_y.view(1, 1, 3, 3)

sobel_conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
sobel_conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

sobel_conv_x.weight = nn.Parameter(sobel_x)
sobel_conv_y.weight = nn.Parameter(sobel_y)


def edge_loss(attn_map, mask, iou):
    loss_ = 0

    mask_clone = mask.clone()[1:-1, 1:-1]

    attn_map_clone = attn_map.unsqueeze(0).unsqueeze(0)
    attn_map_clone = attn_map_clone / attn_map_clone.max().detach()
    attn_map_clone = F.pad(attn_map_clone, (1, 1, 1, 1), mode='reflect')
    attn_map_clone = smth_3(attn_map_clone)

    sobel_output_x = sobel_conv_x(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_output_y = sobel_conv_y(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_sum = torch.sqrt(sobel_output_y ** 2 + sobel_output_x ** 2)
    sobel_sum = sobel_sum

    loss_ += 1 - (sobel_sum * mask_clone).sum() / sobel_sum.sum() * (1 - iou)

    return loss_


def compute_rnb(attn_maps_mid, attn_maps_up, attn_maps_down, attn_self, bboxes, object_positions, height, width,
                iter=None,
                attn_weight=None, ):
    # W = H = dim = 96

    dim_width = int(np.ceil(width / 8))
    dim_height = int(np.ceil(height / 8))

    loss = 0
    object_number = len(bboxes)

    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()

    attn16_list = []
    for attn_map_integrated in attn_maps_up[0]:
        attn16_list.append(attn_map_integrated)

    for attn_map_integrated in attn_maps_down[-1]:
        attn16_list.append(attn_map_integrated)

    attn_all_list = []
    attn_edge = []
    up_div_factor_list = [32, 16, 8]
    for sub_list, div_factor in zip(attn_maps_up, up_div_factor_list):  # div 32,16,8
        for item in sub_list:
            b, i, j = item.shape
            W = int(np.ceil(width / div_factor))
            H = int(np.ceil(height / div_factor))
            # sub_res = int(math.sqrt(i))
            item = item.reshape(b, H, W, j).permute(3, 0, 1, 2).mean(dim=1, keepdim=True)
            if min(W, H) <= 32:
                attn_all_list.append(F.interpolate(item, (dim_height,dim_width), mode='bilinear'))
                attn_edge.append(F.interpolate(item, (dim_height,dim_width), mode='bilinear'))

    down_div_factor_list = [8, 16, 32]
    for sub_list, div_factor in zip(attn_maps_down, down_div_factor_list):
        for item in sub_list:
            b, i, j = item.shape
            W = int(np.ceil(width / div_factor))
            H = int(np.ceil(height / div_factor))
            # sub_res = int(math.sqrt(i))
            item = item.reshape(b, H, W, j).permute(3, 0, 1, 2).mean(dim=1, keepdim=True)
            if min(W, H) <= 32:
                attn_all_list.append(F.interpolate(item, (dim_height,dim_width), mode='bilinear'))

    div_factor = 64
    for item in attn_maps_mid:
        b, i, j = item.shape
        # sub_res = int(math.sqrt(i))
        W = int(np.ceil(width / div_factor))
        H = int(np.ceil(height / div_factor))
        item = item.reshape(b, H, W, j).permute(3, 0, 1, 2).mean(dim=1, keepdim=True)
        attn_all_list.append(F.interpolate(item, (dim_height,dim_width), mode='bilinear'))
        attn_edge.append(F.interpolate(item, (dim_height,dim_width), mode='bilinear'))

    attn_all_list = torch.cat(attn_all_list, dim=1)
    attn_all_list = attn_all_list.mean(dim=1).permute(1, 2, 0)
    attn_all = attn_all_list[:, :, 1:]

    attn_edge = torch.cat(attn_edge, dim=1)
    attn_edge = attn_edge.mean(dim=1).permute(1, 2, 0)
    attn_edge = torch.nn.functional.softmax(attn_edge[:, :, 1:] * 120, dim=-1)

    # H = W = 96  # 64

    obj_loss = 0
    edgeonly_loss = 0

    # rows, cols = torch.meshgrid(torch.arange(dim_height), torch.arange(dim_width))
    # positions = torch.stack([rows.flatten(), cols.flatten()], dim=-1)
    # positions = positions.to(attn_all.device) / dim_height

    # import ipdb; ipdb.set_trace()
    for obj_idx in range(object_number):

        for num, obj_position in enumerate(object_positions[obj_idx]):
            true_obj_position = obj_position - 1
            # print(obj_position)
            if num == 0:
                att_map_obj_raw = attn_all[:, :, true_obj_position]
                att_map_edge = attn_edge[:, :, true_obj_position]

            else:
                att_map_obj_raw = att_map_obj_raw + attn_all[:, :, true_obj_position]
                att_map_edge = att_map_edge + attn_edge[:, :, true_obj_position]

        attn_norm = (att_map_obj_raw - att_map_obj_raw.min()) / (att_map_obj_raw.max() - att_map_obj_raw.min())

        mask = torch.zeros(size=(dim_height, dim_width)).cuda() if torch.cuda.is_available() else torch.zeros(
            size=(dim_height, dim_width)).float()  # ,dtype=att_map_edge.dtype
        mask_clone = mask.clone()

        for obj_box in bboxes[obj_idx]:
            x_min, y_min, x_max, y_max = int(obj_box[0] * dim_width), \
                int(obj_box[1] * dim_height), int(obj_box[2] * dim_width), int(obj_box[3] * dim_height)
            mask[y_min: y_max, x_min: x_max] = 1

        mask_none_cls = (1 - mask)

        threshold = (attn_norm * mask).sum() / mask.sum() / 5 * 2 + \
                    ((attn_norm * mask_none_cls).sum() / mask_none_cls.sum() / 5 * 3) if mask_none_cls.sum() != 0 else 0

        if threshold.isnan():
            import pdb
            pdb.set_trace()
        thres_image = attn_norm.gt(threshold) * 1.0
        noise_image = F.sigmoid(20 * (attn_norm - threshold))

        rows, cols = torch.where(thres_image > 0.3)

        # import pdb
        # pdb.set_trace()
        if rows.size()[0] == 0 or cols.size()[0] == 0:
            print("iter %d, rows.size()[0]:%d, cols.size()[0]:%d" % (iter, rows.size()[0], cols.size()[0]))
            continue
        x1, y1 = cols.min(), rows.min()
        x2, y2 = cols.max(), rows.max()

        mask_aug = mask_clone
        mask_aug[y1: y2, x1: x2] = 1
        mask_aug_in = mask_aug * mask
        iou = (mask_aug * mask).sum() / torch.max(mask_aug, mask).sum()
        # print("iter: %d,threshold: %f, valid: %d,iou: %f" % (iter, threshold, torch.sum(thres_image > 0.3).item(), iou))
        if iou < 0.85:
            this_cls_diff_aug_1 = (mask_aug - attn_norm).detach() + attn_norm
            this_cls_diff_aug_in_1 = (mask_aug_in - attn_norm).detach() + attn_norm

            obj_loss += 1 - (1 - iou) * (mask * this_cls_diff_aug_in_1).sum() * (1 / this_cls_diff_aug_1.sum().detach())
            obj_loss += 1 - (1 - iou) * (mask * this_cls_diff_aug_in_1).sum().detach() * (1 / this_cls_diff_aug_1.sum())

            # if object_number > 1 and obj_idx > -1:
            #     if (att_map_obj_raw * mask).max() < (att_map_obj_raw * (1 - mask)).max():
            #         edgeonly_loss += edge_loss(att_map_edge, mask, iou) * 1
            #         print("iter:%d,edgeonly_loss:%f" % (iter, edgeonly_loss))

            obj_loss += 1 - (1 - iou) * ((mask * noise_image).sum() * (1 / noise_image.sum().detach())) * 0.5
            obj_loss += 1 - (1 - iou) * ((mask * noise_image).sum().detach() * (1 / noise_image.sum())) * 0.5

    loss += obj_loss / object_number
    # print("iter:%d,loss:%f" % (iter, loss))
    return loss, attn_weight, edgeonly_loss
