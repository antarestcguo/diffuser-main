from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

revision = None
non_ema_revision = None
model_path = 'models/stable-diffusion-2'
clip_model_path = 'models/clip/clip-vit-large-patch14'
# model_path = 'models/stable-diffusion-v1-4'


negative_prompt = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, low quality, long neck, frame, text, cut"
# prompt = "A white kitten snuggles on a soft blanket, its fur sleek and eyes gentle. It purrs softly, its body trembling slightly as if to signal its warmth. WARM ART STYLE, masterpiece, photographic, intricate detail, Ultra Detailed hyperrealistic real photo, 8k, high quality, hdr."
my_prompt = "A white kitten sit on a table, lovely, cute, smile, big eyes,WARM ART STYLE masterpiece photographic, intricate detail Ultra Detailed hyperrealistic real photo , 8k high quality hdr"

my_prompt_engineer = "A white kitten sit on a table,the cat is lovely cute smile big eyes,the image is in the style of WARM ART, it is masterpiece photographic,the image is full of intricate detail, it looks like Ultra Detailed hyperrealistic real photo, the painting is in 8k high quality hdr"

base_prompt = "A white kitten sit on a table, lovely, cute, smile, big eyes"
base_prompt_engineer = "A red panda, a raccoon and a racoon dog are sitting on the grass. the image is in the style of WARM ART, it is masterpiece photographic,the image is full of intricate detail, it looks like Ultra Detailed hyperrealistic real photo, the painting is in 8k high quality hdr"
ft_prompt = "the cat is big eyes, big round face, the image is in the style of WARM ART, it is masterpiece photographic,the image is full of intricate detail, it looks like Ultra Detailed hyperrealistic real photo, the painting is in 8k high quality hdr"

tmp_prompt = "A dog"

save_path = 'gen_imgs'
save_prefix = 'dog'
gen_num = 10

# refine use
refine_num = 4
input_img_name_prefix = 'gen_imgs/cat_myprompt_base'
input_img_name = 'gen_imgs/cat_myprompt_base_0.jpg'
strength = 0.8

# refine and merge
input_img_name_1 = 'gen_imgs/cat_myprompt_engin_2.jpg'
input_img_name_2 = 'gen_imgs/cat_myprompt_engin_3.jpg'
