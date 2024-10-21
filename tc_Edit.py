import argparse
import numpy as np
from PIL import Image
import os
from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch
from tc_utils.model_utils import load_textual_embedding
from tc_utils.model_utils import resizeImg
from gen_config import general_config

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str)
parser.add_argument("--model_id", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--unique_identifier", type=str)
parser.add_argument(
    "--textual_inversion_embedding_path",
    type=str,
    default=None,
    required=True,
    help="Path to textual inversion embedding file",
)
# parser.add_argument("--batch_size", type=int, default=4, help="The number of images to generate")
parser.add_argument("--blending_start_percentage", type=float, default=0.25,
                    help="The diffusion steps percentage to jump")
# parser.add_argument("--device", type=str, default="cuda")
# parser.add_argument("--output_path", type=str, default="outputs/res.jpg", help="The destination output path")

args = parser.parse_args()

model_type = args.model_type
model_id = args.model_id
save_path = args.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)


class BlendedLatnetDiffusion:
    def __init__(self, model_id, textual_inversion_embedding_path):
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.unet = pipe.unet
        self.vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        # add by tc_guo for textual inversion
        self.tokenizer, self.text_encoder = load_textual_embedding(file_path=textual_inversion_embedding_path,
                                                                   tokenizer=tokenizer,
                                                                   text_encoder=text_encoder)

    @torch.no_grad()
    def edit_image(
            self,
            image_path,
            mask_path,
            prompts,
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=torch.manual_seed(42),
            blending_percentage=0.25,
    ):
        batch_size = len(prompts)

        image = Image.open(image_path)
        # resize_img = resizeImg(image)
        image = np.array(image)[:, :, :3]
        source_latents = self._image2latent(image)
        latent_mask, org_mask = self._read_mask(mask_path)

        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to("cuda"))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to("cuda"))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to("cuda").half()

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps[
                 int(len(self.scheduler.timesteps) * blending_percentage):
                 ]:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Blending
            noise_source_latents = self.scheduler.add_noise(
                source_latents, torch.randn_like(latents), t
            )
            latents = latents * latent_mask + noise_source_latents * (1 - latent_mask)

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images

    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")
        image = image.half()
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents

    def _read_mask(self, mask_path: str, dest_size=(64, 64)):
        org_mask = Image.open(mask_path).convert("L")
        mask = org_mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[np.newaxis, np.newaxis, ...]
        mask = torch.from_numpy(mask).half().to("cuda")

        return mask, org_mask


bld = BlendedLatnetDiffusion(model_id, args.textual_inversion_embedding_path)
for it_key, it_prompt in general_config.base_prompt_dict_list.items():
    if it_key in general_config.img_base_dict:
        tmp_name = general_config.img_base_dict[it_key]
        if isinstance(tmp_name, list):
            for idx, it_name in enumerate(tmp_name):

                img_name = os.path.join(general_config.img_base_path, it_name)
                mask_name = os.path.join(general_config.img_base_edit_path, it_name)
                if not os.path.exists(mask_name):
                    continue

                input_prompt = it_prompt.replace("unique_identifier",
                                                 args.unique_identifier) + general_config.positive_promot_str
                for i in range(5):
                    image = bld.edit_image(
                        image_path=img_name,
                        mask_path=mask_name,
                        prompts=input_prompt,
                        blending_percentage=args.blending_start_percentage,
                    )

                    save_name = os.path.join(save_path,
                                             "TE_model_type_base_img2img_%s_%s%d_%d.jpg" % (model_type, it_key, idx, i))
                    Image.fromarray(image).save(save_name)

        else:

            img_name = os.path.join(general_config.img_base_path, tmp_name)
            mask_name = os.path.join(general_config.img_base_edit_path, tmp_name)
            if not os.path.exists(mask_name):
                continue
            input_prompt = it_prompt.replace("unique_identifier",
                                             args.unique_identifier) + general_config.positive_promot_str
            for i in range(5):
                image = bld.edit_image(
                    image_path=img_name,
                    mask_path=mask_name,
                    prompts=input_prompt,
                    blending_percentage=args.blending_start_percentage,
                )
                save_name = os.path.join(save_path, "TE_model_type_base_img2img_%s_%s_%d.jpg" % (model_type, it_key, i))
                Image.fromarray(image).save(save_name)

for it_key, it_prompt in general_config.enhance_prompt_dict_list.items():
    if it_key in general_config.img_base_dict:
        tmp_name = general_config.img_base_dict[it_key]
        if isinstance(tmp_name, list):
            for idx, it_name in enumerate(tmp_name):

                img_name = os.path.join(general_config.img_base_path, it_name)
                mask_name = os.path.join(general_config.img_base_edit_path, it_name)
                if not os.path.exists(mask_name):
                    continue
                input_prompt = it_prompt.replace("unique_identifier",
                                                 args.unique_identifier) + general_config.positive_promot_str
                for i in range(5):
                    image = bld.edit_image(
                        image_path=img_name,
                        mask_path=mask_name,
                        prompts=input_prompt,
                        blending_percentage=args.blending_start_percentage,
                    )

                    save_name = os.path.join(save_path,
                                             "TE_model_type_enhance_img2img_%s_%s%d_%d.jpg" % (
                                                 model_type, it_key, idx, i))
                    Image.fromarray(image).save(save_name)
        else:
            img_name = os.path.join(general_config.img_base_path, tmp_name)
            mask_name = os.path.join(general_config.img_base_edit_path, tmp_name)
            if not os.path.exists(mask_name):
                continue
            input_prompt = it_prompt.replace("unique_identifier",
                                             args.unique_identifier) + general_config.positive_promot_str
            for i in range(5):
                image = bld.edit_image(
                    image_path=img_name,
                    mask_path=mask_name,
                    prompts=input_prompt,
                    blending_percentage=args.blending_start_percentage,
                )
                save_name = os.path.join(save_path,
                                         "TE_model_type_enhance_img2img_%s_%s_%d.jpg" % (model_type, it_key, i))
                Image.fromarray(image).save(save_name)
