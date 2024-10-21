from diffusers import StableDiffusionPipeline
import torch
import os

model_path_dict = {
    "TE-robot": "/home/Antares.Guo/train_model/0620_tc_fttextunet_ori_caption",
    "new1-robot": "/home/Antares.Guo/train_model/0625_tc_fttextunet_ori_caption_shuffe5_new1_robot"
}
template_replace_token_dict = {
    "TE-robot": "<TE-robot>",
    "new1-robot": "<new1>+robot",
}
# template_replace_token = "<TE-robot>"
token_path_dict = {
    "TE-robot": "/home/Antares.Guo/train_model/0620_tc_fttextunet_ori_caption",
"new1-robot": "/home/Antares.Guo/train_model/0625_tc_fttextunet_ori_caption_shuffe5_new1_robot"
}


class BasePipeline:
    def __init__(self, model_id):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path_dict[model_id], torch_dtype=torch.float16).to(
            "cuda")

        replace_tokens = template_replace_token_dict[model_id].split("+")
        self.unique_identifier = " ".join(replace_tokens)
        for it_token in replace_tokens:
            file_name = os.path.join(token_path_dict[model_id], f"{it_token}.bin")
            if os.path.exists(file_name):
                self.pipe.load_textual_inversion(token_path_dict[model_id], weight_name=f"{it_token}.bin")


basepipe = BasePipeline("new1-robot")
