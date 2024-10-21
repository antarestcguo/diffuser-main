# 0206 lora, one_caption, new1+robot, select list, textual inversion，未来可以复用
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train_clean_lora.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "<new1>+robot" --initializer_token="orange+robot" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--instance_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0206_clean_text_<new1>_robot" \
#--caption_flag_str "one_caption" \
#--ft_type "text" \
#--only_new_token


#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train_clean_lora.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/0202_clean_text_<new1>_robot" \
#--modifier_token "<new1>+robot" --initializer_token="orange+robot" \
#--scale_lr --hflip --max_train_steps=1000 --lr_warmup_steps=0 \
#--learning_rate=1e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--instance_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0206_clean_textunetlora_new1_robot" \
#--caption_flag_str "one_caption" \
#--ft_type "unet"


CUDA_VISIBLE_DEVICES=4 python tc_inference_scripts/batch_dreambooth_lora.py \
--model_type "textunet" \
--model_id="./models/stable-diffusion-2" \
--lora_path="/data1/tc_guo/train_model/0206_clean_textunetlora_new1_robot" \
--token_path="/data1/tc_guo/train_model/0202_clean_text_<new1>_robot" \
--replace_token "<new1>+robot" \
--save_path="/home/Antares.Guo/tmp_img/0206_clean_textunetlora"
