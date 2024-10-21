

# 复现原来的
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "<new1>+robot" --initializer_token="orange+robot" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--other_dir /data1/tc_guo/data/TE/total_TE \
#--other_file_name /data1/tc_guo/data/TE/shuffle_caption_5_list.txt \
#--instance_file_name /data1/tc_guo/data/TE/shuffle_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0202_reim_text_<new1>_robot" \
#--caption_flag_str "ori_caption" \
#--ft_type "text"


#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/0202_reim_text_<new1>_robot" \
#--modifier_token "<new1>+robot" --initializer_token="orange+robot" \
#--scale_lr --hflip --max_train_steps=500 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--other_dir /data1/tc_guo/data/TE/total_TE \
#--other_file_name /data1/tc_guo/data/TE/shuffle_caption_5_list.txt \
#--instance_file_name /data1/tc_guo/data/TE/shuffle_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0202_reim_textunet_<new1>_robot" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#
#CUDA_VISIBLE_DEVICES=0 python tc_inference_scripts/batch_dreambooth.py \
#--model_type "textunet" \
#--model_id="/data1/tc_guo/train_model/0202_reim_textunet_<new1>_robot" \
#--token_path="/data1/tc_guo/train_model/0202_reim_text_<new1>_robot" \
#--replace_token "<new1>+robot" \
#--save_path="/home/Antares.Guo/tmp_img/0202_reim_textunet"

## 0202 train text 尝试改动code想tc_train.py对齐，对齐结果节本OK
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train_clean.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "<new1>+robot" --initializer_token="orange+robot" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--instance_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0202_clean_text_<new1>_robot" \
#--caption_flag_str "ori_caption" \
#--ft_type "text" \
#--only_new_token
#
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train_clean.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/0202_clean_text_<new1>_robot" \
#--modifier_token "<new1>+robot" --initializer_token="orange+robot" \
#--scale_lr --hflip --max_train_steps=500 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--instance_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0202_clean_textunet_<new1>_robot" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#
#CUDA_VISIBLE_DEVICES=0 python tc_inference_scripts/batch_dreambooth.py \
#--model_type "textunet" \
#--model_id="/data1/tc_guo/train_model/0202_clean_textunet_<new1>_robot" \
#--token_path="/data1/tc_guo/train_model/0202_clean_text_<new1>_robot" \
#--replace_token "<new1>+robot" \
#--save_path="/home/Antares.Guo/tmp_img/0202_clean_textunet"  # 基本和原来的tc_train一致，一点都不能改啊。现在尝试从tc_train改一些

# 只用robot 结果也是OK的，robot没有跑偏！
#CUDA_VISIBLE_DEVICES=0 python tc_inference_scripts/batch_dreambooth.py \
#--model_type "textunet" \
#--model_id="/data1/tc_guo/train_model/0202_clean_textunet_<new1>_robot" \
#--token_path="/data1/tc_guo/train_model/0202_clean_text_<new1>_robot" \
#--replace_token "robot" \
#--save_path="/home/Antares.Guo/tmp_img/0202_clean_textunet_robottest"


# 复现原来的,尝试一些变动，换list影响不大，和one_caption比起来，one_caption影响更大
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "<new1>+robot" --initializer_token="orange+robot" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--other_dir /data1/tc_guo/data/TE/total_TE \
#--other_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--instance_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0205_reim_text_newlist_<new1>_robot" \
#--caption_flag_str "ori_caption" \
#--ft_type "text"
#
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/0205_reim_text_newlist_<new1>_robot" \
#--modifier_token "<new1>+robot" --initializer_token="orange+robot" \
#--scale_lr --hflip --max_train_steps=500 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--other_dir /data1/tc_guo/data/TE/total_TE \
#--other_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--instance_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0205_reim_textunet_newlist_<new1>_robot" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#
#CUDA_VISIBLE_DEVICES=4 python tc_inference_scripts/batch_dreambooth.py \
#--model_type "textunet" \
#--model_id="/data1/tc_guo/train_model/0205_reim_textunet_newlist_<new1>_robot" \
#--token_path="/data1/tc_guo/train_model/0205_reim_text_newlist_<new1>_robot" \
#--replace_token "<new1>+robot" \
#--save_path="/home/Antares.Guo/tmp_img/0205_reim_textunet_newlist"

# 0205 tc_train.py对齐后，对齐结果节本OK，改one_caption，new1，designlist,还不如reim结果
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train_clean.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "<new1>" --initializer_token="robot" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--instance_file_name /data1/tc_guo/data/TE/design_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0205_clean_text_design_<new1>" \
#--caption_flag_str "one_caption" \
#--ft_type "text" \
#--only_new_token
#
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train_clean.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/0205_clean_text_design_<new1>" \
#--modifier_token "<new1>" --initializer_token="robot" \
#--scale_lr --hflip --max_train_steps=500 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--instance_file_name /data1/tc_guo/data/TE/design_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0205_clean_textunet_design_<new1>" \
#--caption_flag_str "one_caption" \
#--ft_type "unet"
#
#
#CUDA_VISIBLE_DEVICES=0 python tc_inference_scripts/batch_dreambooth.py \
#--model_type "textunet" \
#--model_id="/data1/tc_guo/train_model/0205_clean_textunet_design_<new1>" \
#--token_path="/data1/tc_guo/train_model/0205_clean_text_design_<new1>" \
#--replace_token "<new1>" \
#--save_path="/home/Antares.Guo/tmp_img/0205_clean_textunet_design"