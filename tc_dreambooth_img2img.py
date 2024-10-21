from diffusers import StableDiffusionImg2ImgPipeline
import torch
from tc_utils.model_utils import resizeImg
import os
import argparse
from gen_config import general_config
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str)
parser.add_argument("--model_id", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--unique_identifier", type=str)
args = parser.parse_args()

model_type = args.model_type
model_id = args.model_id
save_path = args.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

for it_key, it_prompt in general_config.base_prompt_dict_list.items():
    if it_key in general_config.img_base_dict:
        tmp_name = general_config.img_base_dict[it_key]
        if isinstance(tmp_name, list):
            for idx, it_name in enumerate(tmp_name):

                img_name = os.path.join(general_config.img_base_path, it_name)
                resize_img = resizeImg(Image.open(img_name))
                input_prompt = it_prompt.replace("unique_identifier",
                                                 args.unique_identifier) + general_config.positive_promot_str
                for i in range(5):
                    image = pipe(image=resize_img, prompt=input_prompt,
                                 num_inference_steps=args.num_inference_steps,
                                 guidance_scale=7.5,
                                 negative_prompt=general_config.negative_prompt_str,
                                 strength= 0.6).images[0]
                    image.save(os.path.join(save_path, "TE_model_type_base_img2img_%s_%s%d_%d.jpg" % (model_type, it_key,idx, i)))
        else:
            img_name = os.path.join(general_config.img_base_path, tmp_name)
            resize_img = resizeImg(Image.open(img_name))
            input_prompt = it_prompt.replace("unique_identifier",
                                             args.unique_identifier) + general_config.positive_promot_str
            for i in range(5):
                image = pipe(image=resize_img, prompt=input_prompt,
                             num_inference_steps=args.num_inference_steps,
                             guidance_scale=7.5,
                             negative_prompt=general_config.negative_prompt_str,
                             strength=0.6
                             ).images[0]
                image.save(
                    os.path.join(save_path, "TE_model_type_base_img2img_%s_%s_%d.jpg" % (model_type, it_key, i)))

for it_key, it_prompt in general_config.enhance_prompt_dict_list.items():
    if it_key in general_config.img_base_dict:
        tmp_name = general_config.img_base_dict[it_key]
        if isinstance(tmp_name, list):
            for idx, it_name in enumerate(tmp_name):

                img_name = os.path.join(general_config.img_base_path, it_name)
                resize_img = resizeImg(Image.open(img_name))
                input_prompt = it_prompt.replace("unique_identifier",
                                                 args.unique_identifier) + general_config.positive_promot_str
                for i in range(5):
                    image = pipe(image=resize_img, prompt=input_prompt,
                                 num_inference_steps=args.num_inference_steps,
                                 guidance_scale=7.5,
                                 negative_prompt=general_config.negative_prompt_str,
                                 strength=0.6
                                 ).images[0]
                    image.save(os.path.join(save_path,
                                            "TE_model_type_enhance_img2img_%s_%s%d_%d.jpg" % (model_type, it_key, idx, i)))
        else:
            img_name = os.path.join(general_config.img_base_path, tmp_name)
            resize_img = resizeImg(Image.open(img_name))
            input_prompt = it_prompt.replace("unique_identifier",
                                             args.unique_identifier) + general_config.positive_promot_str
            for i in range(5):
                image = pipe(image=resize_img, prompt=input_prompt,
                             num_inference_steps=args.num_inference_steps,
                             guidance_scale=7.5,
                             negative_prompt=general_config.negative_prompt_str,
                             strength=0.6
                             ).images[0]
                image.save(
                    os.path.join(save_path, "TE_model_type_enhance_img2img_%s_%s_%d.jpg" % (model_type, it_key, i)))

# for it_key, it_prompt in general_config.group_prompt_dict_list.items():
#     for i in range(5):
#         image = pipe(it_prompt + general_config.positive_promot_str, num_inference_steps=args.num_inference_steps,
#                      guidance_scale=7.5,
#                      negative_prompt=general_config.negative_prompt_str
#                      ).images[
#             0]
#         image.save(os.path.join(save_path, "TE_model_type_group_%s_%s_%d.jpg" % (model_type, it_key, i)))
#
# for it_key, it_prompt in general_config.other_prompt_dict_list.items():
#     for i in range(5):
#         image = pipe(it_prompt + general_config.positive_promot_str, num_inference_steps=args.num_inference_steps,
#                      guidance_scale=7.5,
#                      negative_prompt=general_config.negative_prompt_str
#                      ).images[
#             0]
#         image.save(os.path.join(save_path, "TE_model_type_other_%s_%s_%d.jpg" % (model_type, it_key, i)))
