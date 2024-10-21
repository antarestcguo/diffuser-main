from diffusers import StableDiffusionPipeline
import torch
from gen_config import config
import os
import argparse
from gen_config import action_config
from random import choice

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str)
parser.add_argument("--model_id", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--textual_model", type=str)
parser.add_argument("--unique_identifier", type=str)
args = parser.parse_args()

model_type = args.model_type
model_id = args.model_id
save_path = args.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.load_textual_inversion(args.textual_model)

# old type
# prompt_dict = {"test": general_config.test_dict_list,
#                "base": general_config.base_prompt_dict_list,
#                "enhance": general_config.enhance_prompt_dict_list,
#                "group": general_config.group_prompt_dict_list,
#                "other": general_config.other_prompt_dict_list}

prompt_dict = {
    "test": action_config.test_dict_list,
    "action": action_config.action_dict_list,
    "wearing": action_config.wearing_dict_list,
    "group": action_config.group_dict_list,
    "status": action_config.status_dict_list,
    "place": action_config.place_dict_list,
    "scene": action_config.scene_dict_list,
}

for it_type, it_dict in prompt_dict.items():
    for it_key, it_prompt in it_dict.items():
        if isinstance(it_prompt, list):
            it_prompt = choice(it_prompt)
        input_prompt = it_prompt.replace("unique_identifier",
                                         args.unique_identifier) + action_config.positive_promot_str
        for i in range(20):
            image = pipe(input_prompt, num_inference_steps=args.num_inference_steps,
                         guidance_scale=7.5,
                         negative_prompt=action_config.negative_prompt_str
                         ).images[
                0]
            image.save(os.path.join(save_path, "%s_%s_%s_%d.jpg" % (model_type, it_type, it_key, i)))
