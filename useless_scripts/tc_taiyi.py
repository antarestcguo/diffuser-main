from diffusers import StableDiffusionPipeline
import torch
import os

save_path = '../gen_imgs'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# prompt = '甜蜜可口的西瓜，让人感觉非常舒适和愉悦'
# prompt = "切菜板上的西瓜籽"
prompt = "南瓜子放在盘子里，配上一杯油"
torch.backends.cudnn.benchmark = True
pipe = StableDiffusionPipeline.from_pretrained("./models/taiyi/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
pipe.to('cuda')
image = pipe(prompt, guidance_scale=10.0).images[0]

if len(prompt) > 20 or prompt.find('.') != -1:
    result_name = os.path.join(save_path, prompt[:10] + '.jpg')
else:
    result_name = os.path.join(save_path, prompt + '.jpg')

image.save(result_name)
