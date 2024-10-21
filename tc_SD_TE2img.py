from diffusers import StableDiffusionImg2ImgPipeline
import torch
from tc_utils.model_utils import resizeImg
import os
import argparse
from gen_config import dreambooth_config
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str)
parser.add_argument("--model_id", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--num_inference_steps", type=int, default=50)
args = parser.parse_args()

model_type = args.model_type
model_id = args.model_id
save_path = args.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

for it_key, it_prompt in dreambooth_config.base_prompt_dict_list.items():
    img_name = dreambooth_config.TE_img_dict["TE"]
    resize_img = resizeImg(Image.open(img_name))
    for i in range(5):
        image = pipe(image=resize_img, prompt=it_prompt + dreambooth_config.positive_promot_str,
                     num_inference_steps=args.num_inference_steps,
                     guidance_scale=7.5,
                     negative_prompt=dreambooth_config.negative_prompt_str
                     ).images[0]
        image.save(
            os.path.join(save_path, "TE_model_type_base_TE2img_%s_%s_%d.jpg" % (model_type, it_key, i)))

for it_key, it_prompt in dreambooth_config.enhance_prompt_dict_list.items():
    img_name = dreambooth_config.TE_img_dict["TE"]
    resize_img = resizeImg(Image.open(img_name))
    for i in range(5):
        image = pipe(image=resize_img, prompt=it_prompt + dreambooth_config.positive_promot_str,
                     num_inference_steps=args.num_inference_steps,
                     guidance_scale=7.5,
                     negative_prompt=dreambooth_config.negative_prompt_str
                     ).images[0]
        image.save(
            os.path.join(save_path, "TE_model_type_enchance_TE2img_%s_%s_%d.jpg" % (model_type, it_key, i)))

# for it_key, it_prompt in dreambooth_config.group_prompt_dict_list.items():
#     img_name = dreambooth_config.TE_img_dict["TE"]
#     for i in range(5):
#         image = pipe(image=Image.open(img_name), prompt=it_prompt + dreambooth_config.positive_promot_str,
#                      num_inference_steps=args.num_inference_steps,
#                      guidance_scale=7.5,
#                      negative_prompt=dreambooth_config.negative_prompt_str
#                      ).images[0]
#         image.save(
#             os.path.join(save_path, "TE_model_type_group_TE2img_%s_%s_%d.jpg" % (model_type, it_key, i)))
#
# for it_key, it_prompt in dreambooth_config.other_prompt_dict_list.items():
#     img_name = dreambooth_config.TE_img_dict["TE"]
#     for i in range(5):
#         image = pipe(image=Image.open(img_name), prompt=it_prompt + dreambooth_config.positive_promot_str,
#                      num_inference_steps=args.num_inference_steps,
#                      guidance_scale=7.5,
#                      negative_prompt=dreambooth_config.negative_prompt_str
#                      ).images[0]
#         image.save(
#             os.path.join(save_path, "TE_model_type_other_TE2img_%s_%s_%d.jpg" % (model_type, it_key, i)))
