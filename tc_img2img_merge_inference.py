from diffusers import StableDiffusionImg2ImgPipeline
import torch
from gen_config import config
import os
import argparse
from gen_config import action_config
from gen_config import control_config
from tc_utils.utils import setup_seed
from random import choice
import PIL.Image as Image
from tc_utils.model_utils import resizeImg

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str)
parser.add_argument("--model_id", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--token_path", type=str)
parser.add_argument("--attn_path", type=str, default='')
parser.add_argument("--replace_token", type=str)
parser.add_argument("--seed", type=int, default=1024)
parser.add_argument("--bz", type=int, default=1)
parser.add_argument("--gen_num", type=int, default=5)
parser.add_argument("--control_img_dir", type=str)
args = parser.parse_args()

model_type = args.model_type
model_id = args.model_id
save_path = args.save_path

if args.seed is not None:
    setup_seed(args.seed)

if not os.path.exists(save_path):
    os.makedirs(save_path)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
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
# old type
# prompt_dict = {"test": general_config.test_dict_list,
#                "base": general_config.base_prompt_dict_list,
#                "enhance": general_config.enhance_prompt_dict_list,
#                "group": general_config.group_prompt_dict_list,
#                "other": general_config.other_prompt_dict_list}

prompt_dict = {
    # "test": action_config.test_dict_list,
    # "action": action_config.action_dict_list,
    # "wearing": action_config.wearing_dict_list,
    # "group": action_config.group_dict_list,
    # "status": action_config.status_dict_list,
    # "place": action_config.place_dict_list,
    # "scene": action_config.scene_dict_list,

    "action": control_config.action_dict_list,
}

inference_time = int(args.gen_num / args.bz)
for it_type, it_dict in prompt_dict.items():
    # for it_key, it_prompt in it_dict.items():
    #     if isinstance(it_prompt, list):
    #         it_prompt = choice(it_prompt)

    # tmp for control config
    for it_key, it_context in it_dict.items():
        it_prompt = it_context[1]

        input_prompt = it_prompt.replace("unique_identifier",
                                         unique_identifier) + action_config.positive_promot_str
        control_img_name = os.path.join(args.control_img_dir, it_context[0])
        control_img = Image.open(control_img_name)
        control_img = control_img.convert("RGB")
        resize_control_img = resizeImg(control_img)
        for i in range(inference_time):
            if args.bz == 1:
                image = \
                    pipe(image=resize_control_img, prompt=input_prompt, num_inference_steps=args.num_inference_steps,
                         guidance_scale=7.5,
                         negative_prompt=action_config.negative_prompt_str
                         ).images[
                        0]
                image.save(os.path.join(save_path, "%s_%s_%s_%d.jpg" % (model_type, it_type, it_key, i)))
            else:
                image = pipe(image=resize_control_img, prompt=input_prompt,
                             num_inference_steps=args.num_inference_steps,
                             guidance_scale=7.5,
                             negative_prompt=action_config.negative_prompt_str,
                             num_images_per_prompt=args.bz).images
                for idx, it_img in enumerate(image):
                    it_img.save(
                        os.path.join(save_path, "%s_%s_%s_%d.jpg" % (model_type, it_type, it_key, idx + i * args.bz)))
