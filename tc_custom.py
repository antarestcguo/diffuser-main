from diffusers import StableDiffusionPipeline
import torch
from gen_config import config
import os
import argparse
from gen_config import general_config

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
pipe = StableDiffusionPipeline.from_pretrained("./models/stable-diffusion-2", torch_dtype=torch.float16).to("cuda")
pipe.unet.load_attn_procs(model_id, weight_name="pytorch_custom_diffusion_weights.bin")
pipe.load_textual_inversion(model_id, weight_name="<new1>.bin")
pipe.load_textual_inversion(model_id, weight_name="<new2>.bin")
pipe.load_textual_inversion(model_id, weight_name="<new3>.bin")

for it_key, it_prompt in general_config.base_prompt_dict_list.items():
    input_prompt = it_prompt.replace("unique_identifier",
                                     args.unique_identifier) + general_config.positive_promot_str
    for i in range(5):
        image = pipe(input_prompt, num_inference_steps=args.num_inference_steps,
                     guidance_scale=7.5,
                     negative_prompt=general_config.negative_prompt_str
                     ).images[
            0]
        image.save(os.path.join(save_path, "TE_model_type_base_%s_%s_%d.jpg" % (model_type, it_key, i)))

for it_key, it_prompt in general_config.enhance_prompt_dict_list.items():
    input_prompt = it_prompt.replace("unique_identifier",
                                     args.unique_identifier) + general_config.positive_promot_str
    for i in range(5):
        image = pipe(input_prompt, num_inference_steps=args.num_inference_steps,
                     guidance_scale=7.5,
                     negative_prompt=general_config.negative_prompt_str
                     ).images[
            0]
        image.save(os.path.join(save_path, "TE_model_type_enhance_%s_%s_%d.jpg" % (model_type, it_key, i)))

for it_key, it_prompt in general_config.group_prompt_dict_list.items():
    input_prompt = it_prompt.replace("unique_identifier",
                                     args.unique_identifier) + general_config.positive_promot_str
    for i in range(5):
        image = pipe(input_prompt, num_inference_steps=args.num_inference_steps,
                     guidance_scale=7.5,
                     negative_prompt=general_config.negative_prompt_str
                     ).images[
            0]
        image.save(os.path.join(save_path, "TE_model_type_group_%s_%s_%d.jpg" % (model_type, it_key, i)))

for it_key, it_prompt in general_config.other_prompt_dict_list.items():
    input_prompt = it_prompt.replace("unique_identifier",
                                     args.unique_identifier) + general_config.positive_promot_str
    for i in range(5):
        image = pipe(input_prompt, num_inference_steps=args.num_inference_steps,
                     guidance_scale=7.5,
                     negative_prompt=general_config.negative_prompt_str
                     ).images[
            0]
        image.save(os.path.join(save_path, "TE_model_type_other_%s_%s_%d.jpg" % (model_type, it_key, i)))
