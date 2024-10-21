#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train_fixed_inpainting.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2-inpainting" \
#--modifier_token "<new1>+robot" --initializer_token="orange+robot" \
#--scale_lr --hflip --max_train_steps=4000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512 --num_class_images=200  \
#--instance_dir /home/Antares.Guo/data/total_TE/ \
#--mask_dir /home/Antares.Guo/data/total_TE_labelmeMask/ \
#--other_dir /home/Antares.Guo/data/total_TE/ \
#--instance_file_name /home/Antares.Guo/data/caption_total_list_Finpainting.txt \
#--other_file_name /home/Antares.Guo/data/caption_total_list_Finpainting.txt \
#--output_dir="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--caption_flag_str "ori_caption" \
#--ft_type "text"
#
#CUDA_VISIBLE_DEVICES=0 python tc_inpainting_inference.py \
#--model_type="<new1>+robot" \
#--model_id="./models/stable-diffusion-2-inpainting" \
#--save_path="/home/Antares.Guo/tmp_img/0721_tc_fttext_ori_caption_sd2_Finpaint/text_total" \
#--token_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--attn_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--replace_token="<new1>+robot" \
#--seed=1024 \
#--img_dir="/home/Antares.Guo/data/inpainting_template" \
#--mask_dir="/home/Antares.Guo/data/inpainting_template_labelmeMask"

#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train_fixed_inpainting.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2-inpainting" \
#--token_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
#--modifier_token "<new1>+robot" --initializer_token="orange+robot" \
#--scale_lr --hflip --max_train_steps=1000 --lr_warmup_steps=0 \
#--learning_rate=2e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512 --num_class_images=200  \
#--instance_dir /home/Antares.Guo/data/total_TE/ \
#--mask_dir /home/Antares.Guo/data/total_TE_labelmeMask/ \
#--other_dir /home/Antares.Guo/data/total_TE/ \
#--instance_file_name /home/Antares.Guo/data/caption_total_list_Finpainting.txt \
#--other_file_name /home/Antares.Guo/data/caption_total_list_Finpainting.txt \
#--output_dir="/home/Antares.Guo/train_model/0721_tc_loadtextftunet_ori_caption_total_FinpaintL_<new1>+robot" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"

CUDA_VISIBLE_DEVICES=0 python tc_inpainting_inference.py \
--model_type="<new1>+robot" \
--model_id="/home/Antares.Guo/train_model/0721_tc_loadtextftunet_ori_caption_total_Finpaint_<new1>+robot" \
--save_path="/home/Antares.Guo/tmp_img/0721_tc_fttext_ori_caption_sd2_Finpaint/unet_total_blend" \
--token_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
--attn_path="/home/Antares.Guo/train_model/0721_tc_fttext_ori_caption_sd2_Finpaint_<new1>+robot" \
--replace_token="<new1>+robot" \
--seed=1024 \
--img_dir="/home/Antares.Guo/data/inpainting_template" \
--mask_dir="/home/Antares.Guo/data/inpainting_template_labelmeMask" \
--num_inference_steps=150 \
--max_side=1080

