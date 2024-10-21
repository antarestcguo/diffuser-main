from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
import torch
from PIL import Image
import os
import argparse
from gen_config import posefile_control_config
from tc_utils.utils import setup_seed
import tc_utils.controlnet_utils as control_utils
import numpy as np
from controlnet_aux.util import resize_image

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str)
parser.add_argument("--model_id", type=str)
parser.add_argument("--pose_model", type=str)
parser.add_argument("--control_model", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--num_inference_steps", type=int, default=100)
parser.add_argument("--token_path", type=str, default='')
parser.add_argument("--attn_path", type=str, default='')
parser.add_argument("--replace_token", type=str)
parser.add_argument("--seed", type=int, default=1024)
parser.add_argument("--control_img_dir", type=str)
parser.add_argument("--pose_img_dir", type=str, default="")
parser.add_argument("--b_align", action='store_true')
parser.add_argument("--TE_pose_file", type=str)
parser.add_argument("--image_resolution", type=int, default=512)
parser.add_argument("--prompt_type", type=str, default='auto')  # [auto,user,no]
parser.add_argument("--empty_control", action='store_true')
args = parser.parse_args()

model_type = args.model_type
pose_model_name = args.pose_model
control_model_name = args.control_model
model_id = args.model_id
save_path = args.save_path
img_size = 512
if args.seed is not None:
    setup_seed(args.seed)

if not os.path.exists(save_path):
    os.makedirs(save_path)

# read template pose
temp_pose, _, _ = control_utils.read_template_json2KeypointList_new(json_name=args.TE_pose_file)

# load model
model = OpenposeDetector.from_pretrained(pose_model_name)
controlnet = ControlNetModel.from_pretrained(
    control_model_name, torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# load textual inversion files
replace_tokens = args.replace_token.split("+")
unique_identifier = " ".join(replace_tokens)
for it_token in replace_tokens:
    file_name = os.path.join(args.token_path, f"{it_token}.bin")
    if os.path.exists(file_name):
        pipe.load_textual_inversion(args.token_path, weight_name=f"{it_token}.bin")
    else:
        print("no file:", file_name)
file_name = os.path.join(args.attn_path, "pytorch_custom_diffusion_weights.bin")
if os.path.exists(file_name):
    print("load file", file_name)
    pipe.unet.load_attn_procs(args.attn_path, weight_name="pytorch_custom_diffusion_weights.bin")
else:
    print("no file:", file_name)

# prompt to test
prompt_dict = {
    "action": posefile_control_config.action_dict_list,
}

# start to inference
for it_type, it_dict in prompt_dict.items():  # it_context: img_name,pose_name,auto_prompt,user_prompt
    for it_key, it_context in it_dict.items():
        control_img_name = os.path.join(args.control_img_dir, it_context[0])
        s_n, s_e = os.path.splitext(it_context[1])
        pose_json_name = os.path.join(args.pose_img_dir, s_n + '.json')
        print(pose_json_name)

        if args.prompt_type == "no":
            prompt = unique_identifier + ", " + posefile_control_config.positive_promot_str
        elif args.prompt_type == "auto":
            prompt = it_context[2].replace("unique_identifier",
                                           unique_identifier) + posefile_control_config.positive_promot_str
        elif args.prompt_type == "user":
            if it_context[3] == "":
                prompt = it_context[2].replace("unique_identifier",
                                               unique_identifier) + posefile_control_config.positive_promot_str
            else:
                prompt = it_context[3].replace("unique_identifier",
                                               unique_identifier) + posefile_control_config.positive_promot_str

        # read from json or det by openpose
        extract_pose_img = Image.open(control_img_name)
        if os.path.exists(pose_json_name):
            control_poses = control_utils.read_template_json2KeypointList(img_name=control_img_name,
                                                                          json_name=pose_json_name)
        else:
            control_poses = control_utils.control_detect_pose(extract_pose_img, model)
            if len(control_poses) == 0:
                continue
            control_poses = control_poses[0].body.keypoints
        align_scale_temp_pose = control_utils.align_scale(control_poses, temp_pose)

        # control_poses = control_utils.control_detect_pose(extract_pose_img, model)
        # if len(control_poses) == 0:
        #     continue
        # align_scale_temp_pose = control_utils.align_scale(control_poses[0].body.keypoints, temp_pose)

        # align or not
        if args.b_align:
            polar_temp_pose = control_utils.xy2polar_new(align_scale_temp_pose)
            polar_action_pose = control_utils.xy2polar_new(control_poses)
            if polar_temp_pose is None or polar_action_pose is None:
                continue
            # new_polar_pose = control_utils.align_poses_keep_temp(action_pose=polar_action_pose,
            #                                                      temp_pose=polar_temp_pose)
            new_polar_pose = control_utils.align_poses(action_pose=polar_action_pose,
                                                       temp_pose=polar_temp_pose)
            new_xy_pose = control_utils.polar2xy_new(new_polar_pose)

            # align the xy position to bottom of the action xy position
            act_bottom_y = control_utils.find_bottom_y(control_poses)
            new_xy_bottom_y = control_utils.find_bottom_y(new_xy_pose)
            delta_y = act_bottom_y - new_xy_bottom_y
            new_xy_pose = control_utils.align_bias_y(delta_y=delta_y, keypoint_list=new_xy_pose)

        else:
            new_xy_pose = control_poses

        # draw pose
        input_image = np.array(extract_pose_img, dtype=np.uint8)
        # new resize
        img = resize_image(input_image, args.image_resolution)
        H, W, C = img.shape

        tmp_pose_img = control_utils.draw_poses([align_scale_temp_pose], H, W)
        new_img = control_utils.draw_poses([new_xy_pose], H, W)
        ori_img = control_utils.draw_poses([control_poses], H, W)
        # pose image, PIL.Image type
        tmp_pose_im = Image.fromarray(tmp_pose_img)
        new_im = Image.fromarray(new_img)
        ori_im = Image.fromarray(ori_img)

        # save
        tmp_pose_im.save(os.path.join(save_path, "temp_%s_%s_%s.jpg" % (model_type, it_type, it_key)))
        new_im.save(os.path.join(save_path, "align_%s_%s_%s.jpg" % (model_type, it_type, it_key)))
        ori_im.save(os.path.join(save_path, "oriTEMP_%s_%s_%s.jpg" % (model_type, it_type, it_key)))

        if args.empty_control:
            new_im = Image.fromarray(np.zeros_like(new_img))

        # start to inference
        image = pipe(prompt=prompt,
                     image=new_im,
                     num_inference_steps=20,
                     negative_prompt=posefile_control_config.negative_prompt_str).images[0]

        # save
        image.save(os.path.join(save_path, "%s_%s_%s.jpg" % (model_type, it_type, it_key)))
