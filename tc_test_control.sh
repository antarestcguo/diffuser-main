
#CUDA_VISIBLE_DEVICES=0 python tc_control_inference.py \
#--model_type="sks+orange+round+robot" \
#--model_id="/home/Antares.Guo/train_model/0809_tc_ftunetonly_ori_caption_sd15_<new1>+<new2>+<new3>+robot_lr1e-6" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+orange+round+robot" \
#--seed=1024 \
#--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
#--control_model="/home/Antares.Guo/code/diffusers/models/sd-controlnet-openpose" \
#--save_path="/home/Antares.Guo/tmp_img/0809_tc_ftunetonly_ori_caption_sd15_sks+orange+round+robot_lr1e-6_control/align" \
#--control_img_dir="/home/Antares.Guo/data/control_template" \
#--TE_pose_file="/home/Antares.Guo/data/TE_control_template/Image_20230613173130.json" \
#--b_align

#
#CUDA_VISIBLE_DEVICES=4 python tc_control_inference.py \
#--model_type="sks+grey+sloth+plushie" \
#--model_id="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15sloth_sks+grey+sloth+plushie_2e-6lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+grey+sloth+plushie" \
#--seed=2023 \
#--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
#--control_model="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15sloth_sks+grey+sloth+plushie_2e-6lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftcontrolonly_loadunet_sd15sloth_sks+grey+sloth+plushie_2e-6lr1e-5/align_user" \
#--control_img_dir="/home/Antares.Guo/data/control_template" \
#--TE_pose_file="/home/Antares.Guo/data/grey_sloth_plushie/01.json" \
#--b_align --prompt_type="user"
#
#CUDA_VISIBLE_DEVICES=4 python tc_control_inference.py \
#--model_type="sks+grey+sloth+plushie" \
#--model_id="/home/Antares.Guo/train_model/0810_tc_ftunetonly_sloth_sd15_sks+grey+sloth+plushie_lr1e-6" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+grey+sloth+plushie" \
#--seed=1024 \
#--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
#--control_model="/home/Antares.Guo/code/diffusers/models/sd-controlnet-openpose" \
#--save_path="/home/Antares.Guo/tmp_img/0810_tc_ftunetonly_plushie_sd15_sks+grey+sloth+plushie_lr1e-6/noalign_user" \
#--control_img_dir="/home/Antares.Guo/data/control_template" \
#--TE_pose_file="/home/Antares.Guo/data/grey_sloth_plushie/01.json" \
#--prompt_type="user"

#CUDA_VISIBLE_DEVICES=4 python tc_control_inference.py \
#--model_type="sks+mr+potato+head" \
#--model_id="/home/Antares.Guo/code/diffusers/models/mr-potato-head" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+mr+potato+head" \
#--seed=1024 \
#--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
#--control_model="/home/Antares.Guo/code/diffusers/models/sd-controlnet-openpose" \
#--save_path="/home/Antares.Guo/tmp_img/potato/user_noalign" \
#--control_img_dir="/home/Antares.Guo/data/control_template" \
#--TE_pose_file="/home/Antares.Guo/data/potato/2.json" \
#--prompt_type="user"


#CUDA_VISIBLE_DEVICES=4 python tc_control_inference.py \
#--model_type="sks+red+monster+toy" \
#--model_id="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15monster_sks+red+monster+toy_2e-6lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+red+monster+toy" \
#--seed=1024 \
#--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
#--control_model="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15monster_sks+red+monster+toy_2e-6lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftcontrolonly_loadunet_sd15monster_sks+red+monster+toy_2e-6lr1e-5/align_user" \
#--control_img_dir="/home/Antares.Guo/data/control_template" \
#--TE_pose_file="/home/Antares.Guo/data/monster_toy/01.json" \
#--b_align --prompt_type="user"
#
#CUDA_VISIBLE_DEVICES=4 python tc_control_inference.py \
#--model_type="sks+mr+potato+head" \
#--model_id="/home/Antares.Guo/train_model/0818_tc_ftcontrolonly_loadunet_sd15potato_sks+mr+potato+head_lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+mr+potato+head" \
#--seed=2048 \
#--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
#--control_model="/home/Antares.Guo/train_model/0818_tc_ftcontrolonly_loadunet_sd15potato_sks+mr+potato+head_lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/potato_control/no_align" \
#--control_img_dir="/home/Antares.Guo/data/control_template" \
#--TE_pose_file="/home/Antares.Guo/data/potato/2.json" \
#--b_align --prompt_type="no"

#CUDA_VISIBLE_DEVICES=4 python tc_control_inference.py \
#--model_type="sks+orange+round+robot" \
#--model_id="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15TE_sks+orange+round_robot_1e-6lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+orange+round+robot" \
#--seed=1024 \
#--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
#--control_model="/home/Antares.Guo/train_model/0822_tc_ftcontrolonly_loadunet_sd15TE_sks+orange+round_robot_1e-6lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftcontrolonly_loadunet_sd15TE_sks+orange+round_robot_1e-6lr1e-5/align_user" \
#--control_img_dir="/home/Antares.Guo/data/control_template" \
#--TE_pose_file="/home/Antares.Guo/data/TE_control_template/Image_20230613172521.json" \
#--b_align --prompt_type="user"

CUDA_VISIBLE_DEVICES=4 python tc_control_inference.py \
--model_type="yan" \
--model_id="/data1/tc_guo/train_model/1107_tc_fttextunet_sd2_sks+plushie" \
--token_path="/data1/tc_guo/train_model/1107_tc_fttext_sd2_sks+plushie" \
--attn_path="" \
--replace_token="sks+plushie" \
--seed=2048 \
--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
--control_model="/data1/tc_guo/models/models/controlnet-sd21-openpose-diffusers" \
--save_path="/home/Antares.Guo/tmp_img/1110yan/sd2_control" \
--control_img_dir="/home/Antares.Guo/data/control_template" \
--TE_pose_file="/data1/tc_guo/data/yanbird/bird_template.json" \
--b_align --prompt_type="user"