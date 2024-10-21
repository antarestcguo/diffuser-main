from PIL import Image

img_path = '../gen_imgs/甜蜜可口的西瓜，让人感觉非常舒适和愉悦.jpg'
img_path = '../gen_imgs/切菜板上的西瓜籽.jpg'
img_path = '/home/Antares.Guo/data/baidu_雪天的风景/银装素裹得/2023_4_28_14_37_12_00026.jpg'
image = Image.open(img_path).convert('RGB')

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained("./models/blip_models/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("./models/blip_models/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

inputs = processor(image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=77)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)