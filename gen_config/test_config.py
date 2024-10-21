from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

revision = None
non_ema_revision = None
model_path = 'models/stable-diffusion-2'
clip_model_path = 'models/clip/clip-vit-large-patch14'
# model_path = 'models/stable-diffusion-v1-4'


negative_prompt = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, low quality, long neck, frame, text, cut"

my_prompt = "A white kitten sit on a table, lovely, cute, smile, big eyes, 8k, high quality, hdr"

save_path = 'gen_imgs'
save_prefix = 'kitten'
gen_num = 10
