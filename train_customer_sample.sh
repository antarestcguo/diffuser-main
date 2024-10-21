# sdv2.0, <=2  tokens sd only, for bird, use <new1> instead of sks

# try sdv1.5 for controlnet
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
--main_process_port 20011 \
examples/custom_diffusion/tc_train.py \
--pretrained_model_name_or_path="./models/stable-diffusion-v1-5" \
--modifier_token "sks+blue+bird+plushie" --initializer_token="sks+blue+bird+plushie" \
--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
--resolution=512 --num_class_images=200  \
--instance_dir /data1/tc_guo/data/yanbird/yanbird1107/ \
--other_dir /data1/tc_guo/data/yanbird/yanbird1107/ \
--instance_file_name /data1/tc_guo/data/yanbird/yanbird1107.txt \
--other_file_name /data1/tc_guo/data/yanbird/yanbird1107.txt \
--output_dir="/data1/tc_guo/train_model/1114_tc_ftunetonly_sd15_sks+blue+bird_plushie_lr1e-5" \
--caption_flag_str "ori_caption" \
--ft_type "unet"

CUDA_VISIBLE_DEVICES=4 python tc_merge_inference.py \
--model_type="sks+blue+bird+plushie" \
--model_id="/data1/tc_guo/train_model/1114_tc_ftunetonly_sd15_sks+blue+bird_plushie_lr1e-5" \
--save_path="/home/Antares.Guo/tmp_img/1114_tc_ftunetonly_sd15_sks+blue+bird_plushie_lr1e-5" \
--token_path="" \
--attn_path="" \
--replace_token="sks+blue+bird+plushie" \
--seed=1024 \
--bz=5 --gen_num=5 --num_inference_steps=100

# MeiTuan kangaroo 4 tokens
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "sks+yellow+kangaroo+plushie" --initializer_token="sks+yellow+kangaroo+plushie" \
#--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
#--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200  \
#--instance_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--other_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--instance_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--other_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--output_dir="/data1/tc_guo/train_model/1102_tc_ftunetonly_sd2_sks+yellow+kangaroo+plushie_lr1e-5" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#CUDA_VISIBLE_DEVICES=4 python tc_merge_inference.py \
#--model_type="sks+yellow+kangaroo+plushie" \
#--model_id="/data1/tc_guo/train_model/1102_tc_ftunetonly_sd2_sks+yellow+kangaroo+plushie_lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/1102_tc_ftunetonly_sd2_sks+yellow+kangaroo+plushie_lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+yellow+kangaroo+plushie" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100
#
## MeiTuan kangaroo 3 tokens
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "sks+kangaroo+plushie" --initializer_token="sks+kangaroo+plushie" \
#--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
#--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200  \
#--instance_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--other_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--instance_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--other_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--output_dir="/data1/tc_guo/train_model/1102_tc_ftunetonly_sd2_sks+kangaroo+plushie_lr1e-5" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#CUDA_VISIBLE_DEVICES=4 python tc_merge_inference.py \
#--model_type="sks+kangaroo+plushie" \
#--model_id="/data1/tc_guo/train_model/1102_tc_ftunetonly_sd2_sks+kangaroo+plushie_lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/1102_tc_ftunetonly_sd2_sks+kangaroo+plushie_lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+kangaroo+plushie" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100
#
## MeiTuan kangaroo 2 tokens
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "sks+plushie" --initializer_token="sks+plushie" \
#--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
#--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200  \
#--instance_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--other_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--instance_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--other_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--output_dir="/data1/tc_guo/train_model/1102_tc_ftunetonly_sd2_sks+plushie_lr1e-5" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#CUDA_VISIBLE_DEVICES=4 python tc_merge_inference.py \
#--model_type="sks+yellow+kangaroo+plushie" \
#--model_id="/data1/tc_guo/train_model/1102_tc_ftunetonly_sd2_sks+plushie_lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/1102_tc_ftunetonly_sd2_sks+plushie_lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+plushie" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100