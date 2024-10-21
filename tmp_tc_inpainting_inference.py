from diffusers import StableDiffusionInpaintPipeline
import torch
from gen_config import config
import os
import argparse
from gen_config import action_config
from tc_utils.utils import setup_seed
from random import choice
import PIL.Image as Image

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str)
parser.add_argument("--model_id", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--token_path", type=str)
parser.add_argument("--attn_path", type=str, default='')
parser.add_argument("--replace_token", type=str)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--bz", type=int, default=1)
parser.add_argument("--gen_num", type=int, default=5)
args = parser.parse_args()

model_type = args.model_type
model_id = args.model_id
save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.seed is not None:
    setup_seed(args.seed)

img_name = "/home/Antares.Guo/tmp_dog.png"
mask_name = "/home/Antares.Guo/tmp_dog_mask.png"
img_size = 512
init_image = Image.open(img_name).resize((img_size, img_size))
mask_image = Image.open(mask_name).resize((img_size, img_size))

pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
replace_tokens = args.replace_token.split("+")
unique_identifier = " ".join(replace_tokens)
for it_token in replace_tokens:
    file_name = os.path.join(args.token_path, f"{it_token}.bin")
    if os.path.exists(file_name):
        pipe.load_textual_inversion(args.token_path, weight_name=f"{it_token}.bin")
file_name = os.path.join(args.attn_path, "pytorch_custom_diffusion_weights.bin")
if os.path.exists(file_name):
    print("load file", file_name)
    pipe.unet.load_attn_procs(args.attn_path, weight_name="pytorch_custom_diffusion_weights.bin")
else:
    print("no file:", file_name)

prompt = "a unique_identifier sitting on a park bench,"
input_prompt = prompt.replace("unique_identifier", unique_identifier) + action_config.positive_promot_str
# input_prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=input_prompt, image=init_image, mask_image=mask_image, num_inference_steps=100).images[0]
image.save(os.path.join(save_path, "inpainting.jpg"))
