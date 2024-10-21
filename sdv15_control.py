from controlnet_aux import OpenposeDetector
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
import tc_utils.controlnet_utils as control_utils
from tc_utils.utils import setup_seed
from controlnet_aux.util import resize_image
import os

setup_seed(1024)

# model_name = 'control_v11p_sd15_openpose'
# extract_pose_img_name = '/home/Antares.Guo/yoga4.jpeg'
extract_pose_img_name = "/home/Antares.Guo/child_tennis2.jpeg"
# template_pose_img_name = "/home/Antares.Guo/Minions_template.jpeg"
# template_pose_json_name = "/home/Antares.Guo/Minions_template.json"

template_pose_img_name = "/home/Antares.Guo/data/TE_control_template/Image_20230613173130.png"
template_pose_json_name = "/home/Antares.Guo/data/TE_control_template/Image_20230613173130.json"

# SD_model_name = "./models/ControlNet-v1-1/v1-5-pruned.ckpt"
# control_model_name = "./models/ControlNet-v1-1/control_v11p_sd15_openpose.pth"
# total_model_yaml = "./models/ControlNet-v1-1/control_v11p_sd15_openpose.yaml"


a_prompt = "best quality"
# n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality, human, person, man, women, children, child,human head"
n_prompt = "worst quality, poorly drawn, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, low quality,lowres, bad anatomy, bad hands, cropped, worst quality,"
image_resolution = 512
controlnet_model = '/home/Antares.Guo/code/diffusers/models/sd-controlnet-openpose'
openpose_model = '/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts'
SD_model = '/home/Antares.Guo/train_model/0712_tc_fttextunet_ori_caption_sd15_<new1>+robot'
# SD_model = '/home/Antares.Guo/code/diffusers/models/stable-diffusion-v1-5'

model = OpenposeDetector.from_pretrained(openpose_model)
extract_pose_img = Image.open(extract_pose_img_name)

# start to prepare align
poses = control_utils.control_detect_pose(extract_pose_img, model)
polar_action_pose = control_utils.xy2polar(poses[0].body.keypoints)

# read temp infor
temp_pose = control_utils.read_template_json2KeypointList(img_name=template_pose_img_name,
                                                          json_name=template_pose_json_name)
polar_temp_pose = control_utils.xy2polar(temp_pose)

# align pose
new_polar_pose = control_utils.align_poses(action_pose=polar_action_pose, temp_pose=polar_temp_pose)
new_xy_pose = control_utils.polar2xy(new_polar_pose)

# draw pose
input_image = np.array(extract_pose_img, dtype=np.uint8)
H, W, C = input_image.shape

# new resize
img = resize_image(input_image, image_resolution)
H, W, C = img.shape

new_img = control_utils.draw_poses([new_xy_pose], H, W)
ori_img = control_utils.draw_poses([poses[0].body.keypoints], H, W)
TE_img = control_utils.draw_poses([temp_pose], H, W)

new_im = Image.fromarray(new_img)
ori_im = Image.fromarray(ori_img)
TE_im = Image.fromarray(TE_img)

new_im.save("TE_tennis_pose.jpg")
ori_im.save("ori_tennis_pose.jpg")
TE_im.save("TE_ori_pose.jpg")
# cv2.imwrite("ori_tennis_pose.jpg", ori_img)
# cv2.imwrite("minions_tennis_pose.jpg", new_img)

# direct detect
pose_img = model(extract_pose_img)

controlnet = ControlNetModel.from_pretrained(
    controlnet_model, torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    SD_model, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.load_textual_inversion("/home/Antares.Guo/train_model/0712_tc_fttext_ori_caption_sd15_<new1>+robot/",
                            weight_name="<new1>.bin")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

prompt = "A <new1> robot is holding a tennis racket on his hand and playing tennis on Wimbledon tennis tournament, feel the Olympic spirit, A 18mm Wide-angle perspective photo , hyper realistic, summer vibe, sun shine, sunny and bright, 32k, super details, photorealistic,"

# prompt = "A <new1> robot holds a tennis racket,super details, photorealistic,"

# prompt = "A <new1> robot is holding a Starbucks coffee cup in its left hand, prepared to savor the rich aroma of the coffee. The steam rises from the cup, forming delicate wisps in the air. hyper realistic, 32k, super details, photorealistic,"

# prompt = "A <new1> robot is chef, preparing food in the kitchen. hyper realistic, 32k, super details, photorealistic,"

image = pipe(prompt=prompt + ' ' + a_prompt,
             image=new_im,
             num_inference_steps=20,
             negative_prompt=n_prompt).images[0]

image.save('TE_tennis_newalign.jpg')
