from diffusers import StableDiffusionPipeline, StableDiffusionStepPipeline, StableDiffusionImg2ImgPipeline, \
    StableDiffusionImg2ImgStepPipeline, StableDiffusionImageVariationPipeline
import os
from tc_utils import utils
from tc_utils import model_utils
from gen_config import config

from PIL import Image
import torch

if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)

utils.setup_seed(59049)

pipeline = StableDiffusionPipeline.from_pretrained(
    config.model_path
)
pipeline.to("cuda")

# prompt_tokens = utils.split_tokens(config.prompt)
max_length = 77
prompt_embeds = model_utils.text2embedding(pipeline.tokenizer, pipeline.text_encoder,
                                           config.tmp_prompt, max_length=max_length, b_remove_pad=False)
max_length = prompt_embeds.shape[1]
negative_embeds = model_utils.text2embedding(pipeline.tokenizer, pipeline.text_encoder,
                                             config.negative_prompt, max_length=max_length, b_remove_pad=False)
print("embeds len:", max_length, "negative_embeds len:", negative_embeds.shape)

for i in range(config.gen_num):
    img = pipeline(
        # image=Image.open(ori_img),
        # prompt=config.my_prompt,
        # negative_prompt=config.negative_prompt,
        # strength=config.strength,
        # save_path=config.save_path,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        num_images_per_prompt=1).images[0]

    save_name = os.path.join(config.save_path, config.save_prefix + '_%d.jpg' % i)
    img.save(save_name)
