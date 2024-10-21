import sys

sys.path.append('./')
import tc_utils.controlnet_utils as control_utils
import argparse
import os
import PIL.Image as Image
import numpy as np
from controlnet_aux.util import resize_image, HWC3

parser = argparse.ArgumentParser()
parser.add_argument("--ori_file_name", type=str)
parser.add_argument("--dst_file_name", type=str)
parser.add_argument("--pose_dir", type=str)
parser.add_argument("--json_dir", type=str)
parser.add_argument("--img_dir", type=str, default=None)
args = parser.parse_args()

if not os.path.exists(args.pose_dir):
    os.makedirs(args.pose_dir)
with open(args.ori_file_name, 'r') as f_r, open(args.dst_file_name, 'w') as f_w:
    for line in f_r.readlines():
        tokens = line.strip().split('\t')  # img_name, caption

        s_n, s_e = os.path.splitext(tokens[0])
        pose_name = os.path.join(args.pose_dir, tokens[0])
        json_name = os.path.join(args.json_dir, s_n + '.json')
        if not os.path.exists(json_name):
            print("no json file:", json_name)
            continue
        # read json
        pose, H, W = control_utils.read_template_json2KeypointList_new(json_name=json_name)

        # draw pose img
        pose_img = control_utils.draw_poses([pose], H, W)
        pose_im = Image.fromarray(pose_img)
        pose_im.save(pose_name)

        f_w.write(tokens[0] + '\t' + tokens[1] + '\t' + tokens[0] + '\n')

        # save merge img
        if args.img_dir is None:
            continue
        extract_pose_img = Image.open(os.path.join(args.img_dir, tokens[0]))
        input_image = np.array(extract_pose_img, dtype=np.uint8)
        input_image = HWC3(input_image)

        img = resize_image(input_image, resolution=512)
        # H, W, C = img.shape
        merge_img = control_utils.draw_poses_on_img(poses=[pose], img=img)
        merge_im = Image.fromarray(merge_img)
        merge_name = os.path.join(args.pose_dir, s_n + '_merge' + s_e)
        merge_im.save(merge_name)
