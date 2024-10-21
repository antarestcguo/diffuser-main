# textinversion + dreambooth training step
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-v1-5" \
#--modifier_token "<new1>+orange+round+robot" --initializer_token="sks+orange+round+robot" \
#--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
#--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512 --num_class_images=200  \
#--instance_dir /home/Antares.Guo/data/total_TE/ \
#--other_dir /home/Antares.Guo/data/total_TE/ \
#--instance_file_name /home/Antares.Guo/data/TE_select_caption.txt \
#--other_file_name /home/Antares.Guo/data/TE_select_caption.txt \
#--output_dir="/home/Antares.Guo/train_model/0809_tc_fttext_select5_sd15_<new1>+orange+round+robot" \
#--caption_flag_str "ori_caption" \
#--ft_type "text"
#
#CUDA_VISIBLE_DEVICES=4 python tc_merge_inference.py \
#--model_type="<new1>+orange+round+robot" \
#--model_id="./models/stable-diffusion-v1-5" \
#--save_path="/home/Antares.Guo/tmp_img/0809_tc_fttext_select5_sd15_new1+orange+round+robot" \
#--token_path="/home/Antares.Guo/train_model/0809_tc_fttext_select5_sd15_<new1>+orange+round+robot" \
#--replace_token="<new1>+orange+round+robot" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100

#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20011 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-v1-5" \
#--token_path="/home/Antares.Guo/train_model/0809_tc_fttext_select5_sd15_<new1>+orange+round+robot" \
#--modifier_token "<new1>+orange+round+robot" --initializer_token="sks+orange+round+robot" \
#--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
#--learning_rate=1e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512 --num_class_images=200  \
#--instance_dir /home/Antares.Guo/data/total_TE/ \
#--other_dir /home/Antares.Guo/data/total_TE/ \
#--instance_file_name /home/Antares.Guo/data/shuffle_caption_5_list.txt \
#--other_file_name /home/Antares.Guo/data/shuffle_caption_5_list.txt \
#--output_dir="/home/Antares.Guo/train_model/0809_tc_fttextunet_select5_sd15_<new1>+orange+round+robot" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="<new1>+orange+round+robot" \
#--model_id="/home/Antares.Guo/train_model/0809_tc_fttextunet_select5_sd15_<new1>+orange+round+robot" \
#--save_path="/home/Antares.Guo/tmp_img/0809_tc_fttextunet_select5_sd15_new1+orange+round+robot" \
#--token_path="/home/Antares.Guo/train_model/0809_tc_fttext_select5_sd15_<new1>+orange+round+robot" \
#--attn_path="/home/Antares.Guo/train_model/0809_tc_fttext_select5_sd15_<new1>+orange+round+robot" \
#--replace_token="<new1>+orange+round+robot" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100

# only dreambooth train step
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-v1-5" \
#--modifier_token "sks+red+monster+toy" --initializer_token="sks+red+monster+toy" \
#--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
#--learning_rate=2e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512 --num_class_images=200  \
#--instance_dir /home/Antares.Guo/data/monster_toy/ \
#--other_dir /home/Antares.Guo/data/monster_toy/ \
#--instance_file_name /home/Antares.Guo/data/monster_toy.txt \
#--other_file_name /home/Antares.Guo/data/monster_toy.txt \
#--output_dir="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+red+monster+toy_lr2e-6" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+red+monster+toy" \
#--model_id="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+red+monster+toy_lr2e-6" \
#--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftunetonly_monster_sd15_sks+red+monster+toy_lr2e-6" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+red+monster+toy" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-v1-5" \
#--modifier_token "sks+red+monster+toy" --initializer_token="sks+red+monster+toy" \
#--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
#--learning_rate=5e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512 --num_class_images=200  \
#--instance_dir /home/Antares.Guo/data/monster_toy/ \
#--other_dir /home/Antares.Guo/data/monster_toy/ \
#--instance_file_name /home/Antares.Guo/data/monster_toy.txt \
#--other_file_name /home/Antares.Guo/data/monster_toy.txt \
#--output_dir="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+red+monster+toy_lr5e-6" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+red+monster+toy" \
#--model_id="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+red+monster+toy_lr5e-6" \
#--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftunetonly_monster_sd15_sks+red+monster+toy_lr5e-6" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+red+monster+toy" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#--main_process_port 20010 \
#examples/custom_diffusion/tc_train.py \
#--pretrained_model_name_or_path="./models/stable-diffusion-v1/home/Antares.Guo/tmp_img/0202_clean_textunet-5" \
#--modifier_token "sks+red+monster+toy" --initializer_token="sks+red+monster+toy" \
#--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
#--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512 --num_class_images=200  \
#--instance_dir /home/Antares.Guo/data/monster_toy/ \
#--other_dir /home/Antares.Guo/data/monster_toy/ \
#--instance_file_name /home/Antares.Guo/data/monster_toy.txt \
#--other_file_name /home/Antares.Guo/data/monster_toy.txt \
#--output_dir="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+red+monster+toy_lr1e-5" \
#--caption_flag_str "ori_caption" \
#--ft_type "unet"
#
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+red+monster+toy" \
#--model_id="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+red+monster+toy_lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftunetonly_monster_sd15_sks+red+monster+toy_lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+red+monster+toy" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100

# something test
#CUDA_VISIBLE_DEVICES=0 python tc_merge_inference.py \
#--model_type="sks+grey+sloth+plushie" \
#--model_id="/home/Antares.Guo/train_model/0810_tc_ftunetonly_sloth_sd15_sks+grey+sloth+plushie_lr1e-6" \
#--save_path="/home/Antares.Guo/tmp_img/0810_tc_ftunetonly_plushie_sd15_sks+grey+sloth+plushie_lr1e-6/nocontrol" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+grey+sloth+plushie" \
#--seed=1024 \
#--bz=5 --gen_num=5 --num_inference_steps=100

CUDA_VISIBLE_DEVICES=1 python tc_merge_inference.py \
--model_type="TE" \
--model_id="/data1/tc_guo/train_model/0625_tc_fttextunet_ori_caption_shuffe5_new1_robot" \
--save_path="/home/Antares.Guo/tmp_img/TErobot/poster/w512h768" \
--token_path="/data1/tc_guo/train_model/0625_tc_fttextunet_ori_caption_shuffe5_new1_robot" \
--attn_path="" \
--replace_token="<new1>+robot" \
--seed=1024 \
--bz=5 --gen_num=5 --num_inference_steps=100 \
--gen_width=512 --gen_height=768

CUDA_VISIBLE_DEVICES=1 python tc_merge_inference.py \
--model_type="TE" \
--model_id="/data1/tc_guo/train_model/0625_tc_fttextunet_ori_caption_shuffe5_new1_robot" \
--save_path="/home/Antares.Guo/tmp_img/TErobot/poster/w768h512" \
--token_path="/data1/tc_guo/train_model/0625_tc_fttextunet_ori_caption_shuffe5_new1_robot" \
--attn_path="" \
--replace_token="<new1>+robot" \
--seed=1024 \
--bz=5 --gen_num=5 --num_inference_steps=100 \
--gen_width=768 --gen_height=512