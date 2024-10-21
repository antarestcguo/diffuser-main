#CUDA_VISIBLE_DEVICES=0 python tc_inpainting_inference.py \
#--model_type="<new1>+robot" \
#--model_id="/home/Antares.Guo/train_model/0721_tc_loadtextftunet_ori_caption_total_Finpaint_<new1>+robot" \
#--token_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--attn_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--replace_token="<new1>+robot" \
#--seed=1024 \
#--img_dir="/home/Antares.Guo/data/inpainting_template" \
#--mask_dir="/home/Antares.Guo/data/inpainting_template_labelmeMask" \
#--num_inference_steps=150 \
#--max_side=1080 \
#--b_seq_prompt True \
#--b_blended True \
#--save_path="/home/Antares.Guo/tmp_img/0721_tc_fttext_ori_caption_sd2_Finpaint/unet_total_blend_seq"
#
#
#CUDA_VISIBLE_DEVICES=0 python tc_inpainting_inference.py \
#--model_type="<new1>+robot" \
#--model_id="/home/Antares.Guo/train_model/0721_tc_loadtextftunet_ori_caption_total_Finpaint_<new1>+robot" \
#--token_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--attn_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--replace_token="<new1>+robot" \
#--seed=1024 \
#--img_dir="/home/Antares.Guo/data/inpainting_template" \
#--mask_dir="/home/Antares.Guo/data/inpainting_template_labelmeMask" \
#--num_inference_steps=150 \
#--max_side=1080 \
#--b_seq_prompt False \
#--b_blended True \
#--save_path="/home/Antares.Guo/tmp_img/0721_tc_fttext_ori_caption_sd2_Finpaint/unet_total_blend_TE"
#
#
#CUDA_VISIBLE_DEVICES=0 python tc_inpainting_inference.py \
#--model_type="<new1>+robot" \
#--model_id="/home/Antares.Guo/train_model/0721_tc_loadtextftunet_ori_caption_total_Finpaint_<new1>+robot" \
#--token_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--attn_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--replace_token="<new1>+robot" \
#--seed=1024 \
#--img_dir="/home/Antares.Guo/data/inpainting_template" \
#--mask_dir="/home/Antares.Guo/data/inpainting_template_labelmeMask" \
#--num_inference_steps=150 \
#--max_side=1080 \
#--b_seq_prompt True \
#--b_blended False \
#--save_path="/home/Antares.Guo/tmp_img/0721_tc_fttext_ori_caption_sd2_Finpaint/unet_total_inpaint_seq"
#
#CUDA_VISIBLE_DEVICES=0 python tc_inpainting_inference.py \
#--model_type="<new1>+robot" \
#--model_id="/home/Antares.Guo/train_model/0721_tc_loadtextftunet_ori_caption_total_Finpaint_<new1>+robot" \
#--token_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--attn_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--replace_token="<new1>+robot" \
#--seed=1024 \
#--img_dir="/home/Antares.Guo/data/inpainting_template" \
#--mask_dir="/home/Antares.Guo/data/inpainting_template_labelmeMask" \
#--num_inference_steps=150 \
#--max_side=1080 \
#--b_seq_prompt False \
#--b_blended False \
#--save_path="/home/Antares.Guo/tmp_img/0721_tc_fttext_ori_caption_sd2_Finpaint/unet_total_inpaint_TE"


CUDA_VISIBLE_DEVICES=1 python tc_inpainting_inference.py \
--model_type="<new1>+robot" \
--model_id="/home/Antares.Guo/code/diffusers/models/stable-diffusion-2-inpainting_te" \
--token_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
--attn_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
--replace_token="<new1>+robot" \
--seed=1024 \
--img_dir="/home/Antares.Guo/data/inpainting_template" \
--mask_dir="/home/Antares.Guo/data/inpainting_template_labelmeMask" \
--num_inference_steps=150 \
--max_side=1080 \
--b_seq_prompt False \
--b_blended False \
--save_path="/home/Antares.Guo/tmp_img/0721_tc_fttext_ori_caption_sd2_Finpaint/unet_total_inpaint_mergemodel_TE"