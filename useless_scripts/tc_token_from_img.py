from diffusers import StableDiffusionPipeline, StableDiffusionStepPipeline, StableDiffusionImg2ImgPipeline, \
    StableDiffusionImg2ImgStepPipeline, StableDiffusionImg2ImgMergePipeline,StableDiffusionImageVariationPipeline
from transformers import CLIPProcessor, CLIPModel

import os
from tc_utils import utils
from tc_utils import model_utils
from gen_config import config

from PIL import Image
import torch

if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)

utils.setup_seed(59049)

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    config.model_path
)
pipeline.to("cuda")

# clip model
clip_model = CLIPModel.from_pretrained(config.clip_model_path).to("cuda")
processor = CLIPProcessor.from_pretrained(config.clip_model_path)
img_condition = Image.open(config.input_img_name_2).convert('RGB')
img_feature = model_utils.img2embedding(clip_model, processor, img_condition)

# prompt_tokens = utils.split_tokens(config.prompt)
# max_length = 77
# prompt_embeds = model_utils.text2embedding(pipeline.tokenizer, pipeline.text_encoder,
#                                            config.my_prompt_engineer, max_length=max_length, b_remove_pad=False)
# max_length = prompt_embeds.shape[1]
# negative_embeds = model_utils.text2embedding(pipeline.tokenizer, pipeline.text_encoder,
#                                              config.negative_prompt, max_length=max_length, b_remove_pad=False)
# print("embeds len:", max_length, "negative_embeds len:", negative_embeds.shape)

img = pipeline(
    image=Image.open(config.input_img_name_1),
    # prompt=config.my_prompt,
    # negative_prompt=config.negative_prompt,
    # strength=config.strength,
    # save_path=config.save_path,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    num_images_per_prompt=config.gen_num).images

for i, it_img in enumerate(img):
    save_name = os.path.join(config.save_path, config.save_prefix + '_%d.jpg' % i)
    it_img.save(save_name)
