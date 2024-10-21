from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os
import argparse
from gen_config import inpainting_config
from tc_utils.utils import setup_seed

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str)
parser.add_argument("--model_id", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--num_inference_steps", type=int, default=100)
parser.add_argument("--token_path", type=str)
parser.add_argument("--attn_path", type=str, default='')
parser.add_argument("--replace_token", type=str)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--img_dir", type=str)
parser.add_argument("--mask_dir", type=str)
parser.add_argument("--max_side", type=int, default=512)
parser.add_argument("--b_seq_prompt", type=bool, default=True)
args = parser.parse_args()

model_type = args.model_type
model_id = args.model_id
save_path = args.save_path
img_size = 512
if args.seed is not None:
    setup_seed(args.seed)

if not os.path.exists(save_path):
    os.makedirs(save_path)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
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

prompt_dict = {
    # "demo": inpainting_config.demo_dict_list,
    "place": inpainting_config.place_dict_list,
    "action": inpainting_config.action_dict_list,
    "status": inpainting_config.status_dict_list,

}


def resize_img_and_mask(img, mask, max_side=512):
    width, height = img.size
    img_max_size = max(width, height)
    if max_side is not None:
        if img_max_size > max_side:
            ratio = max_side / img_max_size
            if img_max_size == width:
                new_width = max_side
                new_height = int(height * ratio)
            elif img_max_size == height:
                new_height = max_side
                new_width = int(width * ratio)
        else:
            new_width = width
            new_height = height
    else:
        new_width = width
        new_height = height

    new_width = new_width // 8 * 8
    new_height = new_height // 8 * 8
    new_img = img.resize((new_width, new_height))
    new_mask = mask.resize((new_width, new_height))
    return new_img, new_mask, width, height, new_width, new_height


for it_type, it_dict in prompt_dict.items():
    for it_key, it_context in it_dict.items():
        img_name = os.path.join(args.img_dir, it_context[0])
        mask_name = os.path.join(args.mask_dir, it_context[1])
        it_prompt = it_context[2]
        if (not os.path.exists(img_name)) or (not os.path.exists(mask_name)):
            continue

        # init_image = Image.open(img_name).resize((img_size, img_size))
        # mask_image = Image.open(mask_name).resize((img_size, img_size))

        init_image = Image.open(img_name)
        mask_image = Image.open(mask_name)

        new_img, new_mask, width, height, new_width, new_height = resize_img_and_mask(img=init_image, mask=mask_image,
                                                                                      max_side=args.max_side if args.max_side > 0 else None)
        if args.b_seq_prompt:
            input_prompt = it_prompt.replace("unique_identifier",
                                             unique_identifier) + inpainting_config.positive_promot_str
        else:
            input_prompt = "a unique_identifier," + inpainting_config.positive_promot_str

        image = \
            pipe.blended_edit(prompt=input_prompt, image=new_img, mask_image=new_mask, width=new_width,
                              height=new_height,
                              num_inference_steps=args.num_inference_steps).images[0]
        image.save(os.path.join(save_path, "%s_%s_%s.jpg" % (model_type, it_type, it_key)))
        torch.cuda.empty_cache()
