# sdv2.0, 2 tokens. for bird, use <new1> instead of sks
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "sks+bird" --initializer_token="sks+bird" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200  \
#--instance_dir /data1/tc_guo/data/yanbird/yanbirdplushie/ \
#--other_dir /data1/tc_guo/data/yanbird/yanbirdplushie/ \
#--instance_file_name /data1/tc_guo/data/yanbird/yanbirdplushie.txt \
#--other_file_name /data1/tc_guo/data/yanbird/yanbirdplushie.txt \
#--output_dir="/data1/tc_guo/train_model/1027_tc_fttext_sd2_sks+bird" \
#--caption_flag_str "ori_caption" \
#--ft_type "text"
#
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+bird" \
#--model_id="./models/stable-diffusion-2" \
#--save_path="/home/Antares.Guo/tmp_img/1027_tc_fttext_sd2_sks+bird" \
#--token_path="/data1/tc_guo/train_model/1027_tc_fttext_sd2_sks+bird" \
#--replace_token="sks+bird" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/1027_tc_fttext_sd2_sks+bird" \
#--modifier_token "sks+bird" --initializer_token="sks+bird" \
#--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512 --num_class_images=200  \
#--instance_dir /data1/tc_guo/data/yanbird/yanbirdplushie/ \
#--other_dir /data1/tc_guo/data/yanbird/yanbirdplushie/ \
#--instance_file_name /data1/tc_guo/data/yanbird/yanbirdplushie.txt \
#--other_file_name /data1/tc_guo/data/yanbird/yanbirdplushie.txt \
#--output_dir="/data1/tc_guo/train_model/1027_tc_fttextunet_sd2_sks+bird" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+bird" \
#--model_id="/data1/tc_guo/train_model/1027_tc_fttextunet_sd2_sks+bird" \
#--save_path="/home/Antares.Guo/tmp_img/1027_tc_fttextunet_sd2_sks+bird" \
#--token_path="/data1/tc_guo/train_model/1027_tc_fttext_sd2_sks+bird" \
#--attn_path="/data1/tc_guo/train_model/1027_tc_fttextunet_sd2_sks+bird" \
#--replace_token="sks+bird" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100


# MeiTuan kangaroo bug
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "sks+plushie" --initializer_token="sks+plushie" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200  \
#--instance_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--other_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--instance_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--other_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--output_dir="/data1/tc_guo/train_model/1102_tc_fttext_sd2_sks+plushie" \
#--caption_flag_str "ori_caption" \
#--ft_type "text"
#
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+plushie" \
#--model_id="./models/stable-diffusion-2" \
#--save_path="/home/Antares.Guo/tmp_img/1102_tc_fttext_sd2_sks+plushie" \
#--token_path="/data1/tc_guo/train_model/1102_tc_fttext_sd2_sks+plushie" \
#--replace_token="sks+plushie" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/1027_tc_fttext_sd2_sks+plushie" \
#--modifier_token "sks+plushie" --initializer_token="sks+plushie" \
#--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512 --num_class_images=200  \
#--instance_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--other_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--instance_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--other_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--output_dir="/data1/tc_guo/train_model/1027_tc_fttextunet_sd2_sks+plushie" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+plushie" \
#--model_id="/data1/tc_guo/train_model/1027_tc_fttextunet_sd2_sks+plushie" \
#--save_path="/home/Antares.Guo/tmp_img/1027_tc_fttextunet_sd2_sks+plushie" \
#--token_path="/data1/tc_guo/train_model/1027_tc_fttext_sd2_sks+plushie" \
#--attn_path="/data1/tc_guo/train_model/1027_tc_fttextunet_sks+plushie" \
#--replace_token="sks+plushie" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100

# yanbird sks_plushie
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
--main_process_port 20010 \
examples/custom_diffusion/tc_train.py \
--pretrained_model_name_or_path="./models/stable-diffusion-2" \
--modifier_token "sks+plushie" --initializer_token="bird+plushie" \
--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
--resolution=768 --num_class_images=200  \
--instance_dir /data1/tc_guo/data/yanbird/yanbird1107/ \
--other_dir /data1/tc_guo/data/yanbird/yanbird1107/ \
--instance_file_name /data1/tc_guo/data/yanbird/yanbird1107.txt \
--other_file_name /data1/tc_guo/data/yanbird/yanbird1107.txt \
--output_dir="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+plushie" \
--caption_flag_str "ori_caption" \
--ft_type "text"

CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
--model_type="sks+plushie" \
--model_id="./models/stable-diffusion-2" \
--save_path="/home/Antares.Guo/tmp_img/1107_tc_fttext_sd2_sks+plushie" \
--token_path="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+plushie" \
--replace_token="sks+plushie" \
--seed=1024 \
--bz=5 --gen_num=5 --num_inference_steps=100

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
--main_process_port 20010 \
examples/custom_diffusion/tc_train.py \
--pretrained_model_name_or_path="./models/stable-diffusion-2" \
--token_path="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+plushie" \
--modifier_token "sks+plushie" --initializer_token="bird+plushie" \
--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
--resolution=512 --num_class_images=200  \
--instance_dir /data1/tc_guo/data/yanbird/yanbird1107/ \
--other_dir /data1/tc_guo/data/yanbird/yanbird1107/ \
--instance_file_name /data1/tc_guo/data/yanbird/yanbird1107.txt \
--other_file_name /data1/tc_guo/data/yanbird/yanbird1107.txt \
--output_dir="/data1/tc_guo/train_model/1107_tc_fttextunet_sd2_sks+plushie" \
--caption_flag_str "ori_caption" \
--ft_type "unet"

CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
--model_type="sks+plushie" \
--model_id="/data1/tc_guo/train_model/1107_tc_fttextunet_sd2_sks+plushie" \
--save_path="/home/Antares.Guo/tmp_img/1107_tc_fttextunet_sd2_sks+plushie" \
--token_path="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+plushie" \
--attn_path="/data1/tc_guo/train_model/1107_tc_fttextunet_sks+plushie" \
--replace_token="sks+plushie" \
--seed=1024 \
--bz=5 --gen_num=5 --num_inference_steps=100


# yanbird sks_bird
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "sks+bird" --initializer_token="sks+bird" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200  \
#--instance_dir /data1/tc_guo/data/yanbird/yanbird1107/ \
#--other_dir /data1/tc_guo/data/yanbird/yanbird1107/ \
#--instance_file_name /data1/tc_guo/data/yanbird/yanbird1107.txt \
#--other_file_name /data1/tc_guo/data/yanbird/yanbird1107.txt \
#--output_dir="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+bird" \
#--caption_flag_str "ori_caption" \
#--ft_type "text"

#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+bird" \
#--model_id="./models/stable-diffusion-2" \
#--save_path="/home/Antares.Guo/tmp_img/1107_tc_fttext_sd2_sks+bird" \
#--token_path="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+bird" \
#--replace_token="sks+bird" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+bird" \
#--modifier_token "sks+bird" --initializer_token="sks+bird" \
#--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512 --num_class_images=200  \
#--instance_dir /data1/tc_guo/data/yanbird/yanbird1107/ \
#--other_dir /data1/tc_guo/data/yanbird/yanbird1107/ \
#--instance_file_name /data1/tc_guo/data/yanbird/yanbird1107.txt \
#--other_file_name /data1/tc_guo/data/yanbird/yanbird1107.txt \
#--output_dir="/data1/tc_guo/train_model/1107_tc_fttextunet_sd2_sks+bird" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+bird" \
#--model_id="/data1/tc_guo/train_model/1107_tc_fttextunet_sd2_sks+bird" \
#--save_path="/home/Antares.Guo/tmp_img/1107_tc_fttextunet_sd2_sks+bird" \
#--token_path="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+bird" \
#--attn_path="/data1/tc_guo/train_model/1107_tc_fttextunet_sks+bird" \
#--replace_token="sks+bird" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100

# re tmp MeiTuan
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "sks+plushie" --initializer_token="sks+plushie" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200  \
#--instance_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--other_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--instance_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--other_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--output_dir="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+plushie" \
#--caption_flag_str "ori_caption" \
#--ft_type "text"
#
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+plushie" \
#--model_id="./models/stable-diffusion-2" \
#--save_path="/home/Antares.Guo/tmp_img/1107_tc_fttext_sd2_sks+plushie" \
#--token_path="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+plushie" \
#--replace_token="sks+plushie" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+plushie" \
#--modifier_token "sks+plushie" --initializer_token="sks+plushie" \
#--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512 --num_class_images=200  \
#--instance_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--other_dir /data1/tc_guo/data/MeiTuan/kangaroo1102/ \
#--instance_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--other_file_name /data1/tc_guo/data/MeiTuan/kangaroo1102.txt \
#--output_dir="/data1/tc_guo/train_model/1107_tc_fttextunet_sd2_sks+plushie" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+plushie" \
#--model_id="/data1/tc_guo/train_model/1107_tc_fttextunet_sd2_sks+plushie" \
#--save_path="/home/Antares.Guo/tmp_img/1107_tc_fttextunet_sd2_sks+plushie" \
#--token_path="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+plushie" \
#--attn_path="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+plushie" \
#--replace_token="sks+plushie" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100