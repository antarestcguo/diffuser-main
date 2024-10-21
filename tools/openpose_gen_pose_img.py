from controlnet_aux import OpenposeDetector
import argparse
import PIL.Image as Image
import os
import sys
from controlnet_aux.util import resize_image, HWC3
import numpy as np
import json

sys.path.append('./')
import tc_utils.controlnet_utils as control_utils

parser = argparse.ArgumentParser()
parser.add_argument("--control_img_dir", type=str)
parser.add_argument("--control_pose_dir", type=str)
args = parser.parse_args()

save_path = args.control_pose_dir
if not os.path.exists(save_path):
    os.makedirs(save_path)
control_img_list = os.listdir(args.control_img_dir)
pose_model_name = "/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts"
# load model
model = OpenposeDetector.from_pretrained(pose_model_name)

for it in control_img_list:
    s_n, s_e = os.path.splitext(it)
    if s_e not in ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG']:
        continue
    control_img_name = os.path.join(args.control_img_dir, it)
    print(control_img_name)

    extract_pose_img = Image.open(control_img_name)
    control_poses = control_utils.control_detect_pose(extract_pose_img, model)

    input_image = np.array(extract_pose_img, dtype=np.uint8)
    input_image = HWC3(input_image)

    img = resize_image(input_image, resolution=512)
    H, W, C = img.shape

    if len(control_poses) == 0:
        print("len(control_poses),", len(control_poses))
        continue
    ori_img = control_utils.draw_poses([control_poses[0].body.keypoints], H, W)
    ori_im = Image.fromarray(ori_img)
    ori_im.save(os.path.join(save_path, it))

    # save merge img
    merge_img = control_utils.draw_poses_on_img(poses=[control_poses[0].body.keypoints], img=img)
    merge_im = Image.fromarray(merge_img)
    merge_name = os.path.join(save_path, s_n + '_merge' + s_e)
    merge_im.save(merge_name)


