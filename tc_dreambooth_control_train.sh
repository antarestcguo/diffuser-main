#  potato control
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu \
#--main_process_port 20011 \
#examples/controlnet/train_dreambooth_controlnet.py \
#--pretrained_model_name_or_path="/home/Antares.Guo/code/diffusers/models/mr-potato-head" \
#--controlnet_model_name_or_path="/home/Antares.Guo/code/diffusers/models/sd-controlnet-openpose" \
#--tracker_project_name="controlnet-demo" \
#--modifier_token "sks+mr+potato+head" \
#--scale_lr  --max_train_steps=800 --lr_warmup_steps=0 \
#--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512  \
#--instance_dir /home/Antares.Guo/data/control_imgs/ \
#--control_dir /home/Antares.Guo/data/control_imgs_pose/ \
#--instance_file_name /home/Antares.Guo/data/control_imgs_caption_pose.txt \
#--output_dir="/home/Antares.Guo/train_model/0818_tc_ftcontrolonly_loadunet_sd15potato_sks+mr+potato+head_lr1e-5" \
#--caption_flag_str "ori_caption" \
#--ft_type "control"
#
#
#CUDA_VISIBLE_DEVICES=4 python tc_control_inference.py \
#--model_type="sks+mr+potato+head" \
#--model_id="/home/Antares.Guo/train_model/0818_tc_ftcontrolonly_loadunet_sd15potato_sks+mr+potato+head_lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+mr+potato+head" \
#--seed=1024 \
#--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
#--control_model="/home/Antares.Guo/train_model/0818_tc_ftcontrolonly_loadunet_sd15potato_sks+mr+potato+head_lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/0818_tc_ftcontrolonly_loadunet_sd15potato_sks+mr+potato+head_lr1e-5/" \
#--control_img_dir="/home/Antares.Guo/data/control_template" \
#--TE_pose_file="/home/Antares.Guo/data/potato/2.json" \
#--b_align

# sloth control
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu \
#--main_process_port 20011 \
#examples/controlnet/train_dreambooth_controlnet.py \
#--pretrained_model_name_or_path="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+grey+sloth+plushie_lr2e-6" \
#--controlnet_model_name_or_path="/home/Antares.Guo/code/diffusers/models/sd-controlnet-openpose" \
#--tracker_project_name="controlnet-demo" \
#--modifier_token "sks+grey+sloth+plushie" \
#--scale_lr  --max_train_steps=800 --lr_warmup_steps=0 \
#--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512  \
#--instance_dir /home/Antares.Guo/data/control_imgs/ \
#--control_dir /home/Antares.Guo/data/control_imgs_pose/ \
#--instance_file_name /home/Antares.Guo/data/control_imgs_caption_pose.txt \
#--output_dir="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15sloth_sks+grey+sloth+plushie_2e-6lr1e-5" \
#--caption_flag_str "ori_caption" \
#--ft_type "control"
#
#
#CUDA_VISIBLE_DEVICES=4 python tc_control_inference.py \
#--model_type="sks+grey+sloth+plushie" \
#--model_id="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15sloth_sks+grey+sloth+plushie_2e-6lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+grey+sloth+plushie" \
#--seed=1024 \
#--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
#--control_model="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15sloth_sks+grey+sloth+plushie_2e-6lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftcontrolonly_loadunet_sd15sloth_sks+grey+sloth+plushie_2e-6lr1e-5/" \
#--control_img_dir="/home/Antares.Guo/data/control_template" \
#--TE_pose_file="/home/Antares.Guo/data/grey_sloth_plushie/01.json" \
#--b_align

# monster control
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu \
#--main_process_port 20011 \
#examples/controlnet/train_dreambooth_controlnet.py \
#--pretrained_model_name_or_path="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+red+monster+toy_lr2e-6" \
#--controlnet_model_name_or_path="/home/Antares.Guo/code/diffusers/models/sd-controlnet-openpose" \
#--tracker_project_name="controlnet-demo" \
#--modifier_token "sks+red+monster+toy" \
#--scale_lr  --max_train_steps=800 --lr_warmup_steps=0 \
#--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512  \
#--instance_dir /home/Antares.Guo/data/control_imgs/ \
#--control_dir /home/Antares.Guo/data/control_imgs_pose/ \
#--instance_file_name /home/Antares.Guo/data/control_imgs_caption_pose.txt \
#--output_dir="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15monster_sks+red+monster+toy_2e-6lr1e-5" \
#--caption_flag_str "ori_caption" \
#--ft_type "control"
#
#
#CUDA_VISIBLE_DEVICES=4 python tc_control_inference.py \
#--model_type="sks+red+monster+toy" \
#--model_id="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15monster_sks+red+monster+toy_2e-6lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+red+monster+toy" \
#--seed=1024 \
#--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
#--control_model="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15monster_sks+red+monster+toy_2e-6lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftcontrolonly_loadunet_sd15monster_sks+red+monster+toy_2e-6lr1e-5/" \
#--control_img_dir="/home/Antares.Guo/data/control_template" \
#--TE_pose_file="/home/Antares.Guo/data/monster_toy/01.json" \
#--b_align

# TE control
#CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu \
#--main_process_port 20011 \
#examples/controlnet/train_dreambooth_controlnet.py \
#--pretrained_model_name_or_path="/home/Antares.Guo/train_model/0809_tc_ftunetonly_ori_caption_sd15_<new1>+<new2>+<new3>+robot_lr1e-6_select" \
#--controlnet_model_name_or_path="/home/Antares.Guo/code/diffusers/models/sd-controlnet-openpose" \
#--tracker_project_name="controlnet-demo" \
#--modifier_token "sks+orange+round+robot" \
#--scale_lr  --max_train_steps=800 --lr_warmup_steps=0 \
#--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512  \
#--instance_dir /home/Antares.Guo/data/control_imgs/ \
#--control_dir /home/Antares.Guo/data/control_imgs_pose/ \
#--instance_file_name /home/Antares.Guo/data/control_imgs_caption_pose.txt \
#--output_dir="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15TE_sks+orange+round_robot_1e-6lr1e-5" \
#--caption_flag_str "ori_caption" \
#--ft_type "control"


CUDA_VISIBLE_DEVICES=4 python tc_control_inference.py \
--model_type="sks+orange+round+robot" \
--model_id="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15TE_sks+orange+round_robot_1e-6lr1e-5" \
--token_path="" \
--attn_path="" \
--replace_token="sks+orange+round+robot" \
--seed=1024 \
--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
--control_model="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15TE_sks+orange+round_robot_1e-6lr1e-5" \
--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftcontrolonly_loadunet_sd15TE_sks+orange+round_robot_1e-6lr1e-5/" \
--control_img_dir="/home/Antares.Guo/data/control_template" \
--TE_pose_file="/home/Antares.Guo/data/TE_control_template/Image_20230613172521.json" \
--b_align