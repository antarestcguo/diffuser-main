from diffusers import StableDiffusionPipelineLayout, StableDiffusionPipeline
import torch
# from gen_config import config
import os
import argparse

# from gen_config import action_config
# from gen_config import control_config, posefile_control_config
# from tc_utils.utils import setup_seed
# from random import choice

import sys

sys.path.append(os.path.join(os.getcwd(), "../../"))
from tc_utils.utils import setup_seed
from tc_utils.layout_utils import Pharse2idx, draw_box

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default='/data1/tc_guo/models/models/stable-diffusion-2-layout')
parser.add_argument("--save_path", type=str)
parser.add_argument("--seed", type=int, default=1024)
parser.add_argument("--bz", type=int, default=1)
parser.add_argument("--gen_num", type=int, default=5)

parser.add_argument("--token_path", type=str, default='')
parser.add_argument("--replace_token", type=str, default='')

args = parser.parse_args()
model_id = args.model_id
save_path = args.save_path

if args.seed is not None:
    setup_seed(args.seed)

if not os.path.exists(save_path):
    os.makedirs(save_path)

pipe = StableDiffusionPipelineLayout.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
# input_prompt = "A hello kitty toy is playing with a purple ball."
input_prompt = "a dog and a cat sitting on the grass, sunshine"
negative_prompt_str = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, low quality, long neck, frame, text, cut, human, person, man, girl, woman, lady, gentalman, Multiple buildings"
bboxes = [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]]
# phrases = "hello kitty; ball"
phrases = "dog; cat"
num_inference_steps = 20
gen_width = 768
gen_height = 768

positive_promot_str = " 8k, high quality, hdr."
if args.token_path != "":
    input_prompt = "A unique_identifier is playing with a football"
    # input_prompt = "A unique_identifier is shootting the goal, the football leaves its foot, soaring towards the goal, realistic,sports photography,"
    phrases = "unique_identifier; football"
    replace_tokens = args.replace_token.split("+")
    unique_identifier = " ".join(replace_tokens)
    for it_token in replace_tokens:
        file_name = os.path.join(args.token_path, f"{it_token}.bin")
        if os.path.exists(file_name):
            pipe.load_textual_inversion(args.token_path, weight_name=f"{it_token}.bin")
        else:
            print("no file:", file_name)

layout_dict = {
    "layout_type": "RnB",  # RnB or guide
    # guide param
    "loss_scale": 30,
    "loss_threshold": 0.2,
    "max_iter": 5,  # 5
    "max_index_step": int(num_inference_steps * 0.2),
    # RnB param
    "RnB_loss_threshold": 0.000001,

}
inference_time = int(args.gen_num / args.bz)
for i in range(inference_time):
    # TE
    if args.token_path != "":
        input_prompt = input_prompt.replace("unique_identifier",
                                            unique_identifier)  # + positive_promot_str
        phrases = phrases.replace("unique_identifier",
                                  unique_identifier)

    # -1. layout related
    object_positions = Pharse2idx(input_prompt, phrases)
    print(object_positions)
    image = pipe(input_prompt,
                 bbox=bboxes,
                 object_positions=object_positions,
                 layout_dict=layout_dict,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=7.5,
                 negative_prompt=negative_prompt_str,
                 width=gen_width,
                 height=gen_height,
                 num_images_per_prompt=args.bz,
                 ).images

    for idx, it_img in enumerate(image):
        draw_box(pil_img=it_img, bboxes=bboxes, phrases=phrases,
                 save_path=os.path.join(save_path, "%d.jpg" % (idx + i * args.bz)), img_w=gen_width, img_h=gen_height)
