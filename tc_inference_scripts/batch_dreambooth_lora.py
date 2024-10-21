from diffusers import StableDiffusionPipeline
import torch
import os
import argparse
from random import choice

import sys

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))
from tc_utils.utils import setup_seed
from gen_config import config
from gen_config import action_config
from gen_config import control_config, posefile_control_config

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str)
parser.add_argument("--model_id", type=str)
parser.add_argument("--lora_path", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--token_path", type=str)
parser.add_argument("--replace_token", type=str)
parser.add_argument("--seed", type=int, default=1024)
parser.add_argument("--bz", type=int, default=1)
parser.add_argument("--gen_num", type=int, default=5)
parser.add_argument("--gen_width", type=int, default=768)
parser.add_argument("--gen_height", type=int, default=768)
args = parser.parse_args()

model_type = args.model_type
base_model_id = args.model_id
token_path = args.token_path

save_path = args.save_path
replace_token = args.replace_token

if args.seed is not None:
    setup_seed(args.seed)

if not os.path.exists(save_path):
    os.makedirs(save_path)

# prepare token
replace_tokens = args.replace_token.split("+")
unique_identifier = " ".join(replace_tokens)

# load base unet
pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
a = pipe.load_attn_procs(args.lora_path)
# load textual
for it_token in replace_tokens:
    file_name = os.path.join(args.token_path, f"{it_token}.bin")
    if os.path.exists(file_name):
        pipe.load_textual_inversion(args.token_path, weight_name=f"{it_token}.bin")
    else:
        print("no file:", file_name)

pipe = pipe.to("cuda")

prompt_dict = {
    # "test": action_config.test_dict_list,
    # "action": action_config.action_dict_list,
    # "wearing": action_config.wearing_dict_list,
    # "group": action_config.group_dict_list,
    # "status": action_config.status_dict_list,
    "place": action_config.place_dict_list,
    # "scene": action_config.scene_dict_list,

    # "action": posefile_control_config.action_dict_list,
}

inference_time = int(args.gen_num / args.bz)
for it_type, it_dict in prompt_dict.items():
    for it_key, it_prompt in it_dict.items():
        if isinstance(it_prompt, list):
            it_prompt = choice(it_prompt)

        input_prompt = it_prompt.replace("unique_identifier",
                                         unique_identifier) + action_config.positive_promot_str

        for i in range(inference_time):
            if args.bz == 1:
                # T1 = time.clock()
                image = pipe(input_prompt, num_inference_steps=args.num_inference_steps,
                             guidance_scale=7.5,
                             negative_prompt=action_config.negative_prompt_str,
                             width=args.gen_width,
                             height=args.gen_height,
                             ).images[
                    0]
                # T2 = time.clock()
                # time_list.append((T2 - T1) * 1000)
                image.save(os.path.join(save_path, "%s_%s_%s_%d.jpg" % (model_type, it_type, it_key, i)))
            else:
                image = pipe(input_prompt, num_inference_steps=args.num_inference_steps,
                             guidance_scale=7.5,
                             negative_prompt=action_config.negative_prompt_str,
                             num_images_per_prompt=args.bz,
                             width=args.gen_width,
                             height=args.gen_height,
                             ).images
                for idx, it_img in enumerate(image):
                    it_img.save(
                        os.path.join(save_path, "%s_%s_%s_%d.jpg" % (model_type, it_type, it_key, idx + i * args.bz)))
