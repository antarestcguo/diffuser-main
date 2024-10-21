# 复现原来的,尝试一些变动，使用原来list，但是使用one_caption，换one_caption影响更大
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
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
#--output_dir="/data1/tc_guo/train_model/0205_reim_text_onecaption_<new1>_robot" \
#--caption_flag_str "one_caption" \
#--ft_type "text"
#
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/0205_reim_text_onecaption_<new1>_robot" \
#--modifier_token "<new1>+robot" --initializer_token="orange+robot" \
#--scale_lr --hflip --max_train_steps=500 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--other_dir /data1/tc_guo/data/TE/total_TE \
#--other_file_name /data1/tc_guo/data/TE/shuffle_caption_5_list.txt \
#--instance_file_name /data1/tc_guo/data/TE/shuffle_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0205_reim_textunet_onecaption_<new1>_robot" \
#--caption_flag_str "one_caption" \
#--ft_type "unet"
#
#
#CUDA_VISIBLE_DEVICES=0 python tc_inference_scripts/batch_dreambooth.py \
#--model_type "textunet" \
#--model_id="/data1/tc_guo/train_model/0205_reim_textunet_onecaption_<new1>_robot" \
#--token_path="/data1/tc_guo/train_model/0205_reim_text_onecaption_<new1>_robot" \
#--replace_token "<new1>+robot" \
#--save_path="/home/Antares.Guo/tmp_img/0205_reim_textunet_onecaption"

# 只用robot 没有带跑偏！！！！nice！！！
#CUDA_VISIBLE_DEVICES=0 python tc_inference_scripts/batch_dreambooth.py \
#--model_type "textunet" \
#--model_id="/data1/tc_guo/train_model/0205_reim_textunet_onecaption_<new1>_robot" \
#--token_path="/data1/tc_guo/train_model/0205_reim_text_onecaption_<new1>_robot" \
#--replace_token "robot" \
#--save_path="/home/Antares.Guo/tmp_img/0205_reim_textunet_onecaption_robottest"

## 新list，one caption，换只有new1
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "<new1>" --initializer_token="robot" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--other_dir /data1/tc_guo/data/TE/total_TE \
#--other_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--instance_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0205_reim_text_onecaption_newlist_<new1>" \
#--caption_flag_str "one_caption" \
#--ft_type "text"
#
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/0205_reim_text_onecaption_newlist_<new1>" \
#--modifier_token "<new1>" --initializer_token="robot" \
#--scale_lr --hflip --max_train_steps=500 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--other_dir /data1/tc_guo/data/TE/total_TE \
#--other_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--instance_file_name /data1/tc_guo/data/TE/select_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0205_reim_textunet_onecaption_newlist_<new1>" \
#--caption_flag_str "one_caption" \
#--ft_type "unet"
#
#
#CUDA_VISIBLE_DEVICES=0 python tc_inference_scripts/batch_dreambooth.py \
#--model_type "textunet" \
#--model_id="/data1/tc_guo/train_model/0205_reim_textunet_onecaption_newlist_<new1>" \
#--token_path="/data1/tc_guo/train_model/0205_reim_text_onecaption_newlist_<new1>" \
#--replace_token "<new1>+robot" \
#--save_path="/home/Antares.Guo/tmp_img/0205_reim_textunet_onecaption_newlist"

# 新design list，one caption，换只有new1,只有new1就影响很多，designlist会把小特变长条
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--modifier_token "<new1>" --initializer_token="robot" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--other_dir /data1/tc_guo/data/TE/total_TE \
#--other_file_name /data1/tc_guo/data/TE/design_caption_5_list.txt \
#--instance_file_name /data1/tc_guo/data/TE/design_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0205_reim_text_onecaption_designlist_<new1>" \
#--caption_flag_str "one_caption" \
#--ft_type "text"
#
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-2" \
#--token_path="/data1/tc_guo/train_model/0205_reim_text_onecaption_designlist_<new1>" \
#--modifier_token "<new1>" --initializer_token="robot" \
#--scale_lr --hflip --max_train_steps=500 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=768 --num_class_images=200 \
#--instance_dir /data1/tc_guo/data/TE/total_TE \
#--other_dir /data1/tc_guo/data/TE/total_TE \
#--other_file_name /data1/tc_guo/data/TE/design_caption_5_list.txt \
#--instance_file_name /data1/tc_guo/data/TE/design_caption_5_list.txt \
#--output_dir="/data1/tc_guo/train_model/0205_reim_textunet_onecaption_designlist_<new1>" \
#--caption_flag_str "one_caption" \
#--ft_type "unet"
#
#
#CUDA_VISIBLE_DEVICES=0 python tc_inference_scripts/batch_dreambooth.py \
#--model_type "textunet" \
#--model_id="/data1/tc_guo/train_model/0205_reim_textunet_onecaption_designlist_<new1>" \
#--token_path="/data1/tc_guo/train_model/0205_reim_text_onecaption_designlist_<new1>" \
#--replace_token "<new1>+robot" \
#--save_path="/home/Antares.Guo/tmp_img/0205_reim_textunet_onecaption_designlist"