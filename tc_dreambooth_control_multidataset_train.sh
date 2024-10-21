#  potato control
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu \
#--main_process_port 20010 \
#examples/controlnet/train_dreambooth_multidataset_controlnet.py \
#--pretrained_model_name_or_path="/home/Antares.Guo/code/diffusers/models/mr-potato-head" \
#--controlnet_model_name_or_path="/home/Antares.Guo/code/diffusers/models/sd-controlnet-openpose" \
#--tracker_project_name="controlnet-demo" \
#--modifier_token "sks+mr+potato+head" \
#--scale_lr  --max_train_steps=200 --lr_warmup_steps=0 \
#--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
#--resolution=512  \
#--instance_dir /home/Antares.Guo/data/potato/ \
#--control_dir /home/Antares.Guo/data/potato_pose/ \
#--instance_file_name /home/Antares.Guo/data/potato_caption_posecontrol.txt \
#--other_instance_dir /home/Antares.Guo/data/control_imgs/ \
#--other_control_dir /home/Antares.Guo/data/control_imgs_pose/ \
#--other_instance_file_name /home/Antares.Guo/data/control_imgs_caption_pose.txt \
#--output_dir="/home/Antares.Guo/train_model/0819_tc_ftcontrol_loadunet_sd15potato_sks+mr+potato+head_multi_lr1e-5" \
#--caption_flag_str "ori_caption" \
#--ft_type "control"
#
#
#CUDA_VISIBLE_DEVICES=0 python tc_control_inference.py \
#--model_type="sks+mr+potato+head" \
#--model_id="/home/Antares.Guo/train_model/0819_tc_ftcontrol_loadunet_sd15potato_sks+mr+potato+head_multi_lr1e-5" \
#--token_path="" \
#--attn_path="" \
#--replace_token="sks+mr+potato+head" \
#--seed=1024 \
#--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
#--control_model="/home/Antares.Guo/train_model/0819_tc_ftcontrol_loadunet_sd15potato_sks+mr+potato+head_multi_lr1e-5" \
#--save_path="/home/Antares.Guo/tmp_img/0819_tc_ftcontrol_loadunet_sd15potato_sks+mr+potato+head_multi_lr1e-5/" \
#--control_img_dir="/home/Antares.Guo/data/control_template" \
#--TE_pose_file="/home/Antares.Guo/data/potato/2.json" \
#--b_align


#  sloth control
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu \
--main_process_port 20010 \
examples/controlnet/train_dreambooth_multidataset_controlnet.py \
--pretrained_model_name_or_path="/home/Antares.Guo/train_model/0810_tc_ftunetonly_sloth_sd15_sks+grey+sloth+plushie_lr1e-6" \
--controlnet_model_name_or_path="/home/Antares.Guo/code/diffusers/models/sd-controlnet-openpose" \
--tracker_project_name="controlnet-demo" \
--modifier_token "sks+grey+sloth+plushie" \
--scale_lr  --max_train_steps=80 --lr_warmup_steps=0 \
--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
--resolution=512  \
--instance_dir /home/Antares.Guo/data/potato/ \
--control_dir /home/Antares.Guo/data/potato_pose/ \
--instance_file_name /home/Antares.Guo/data/potato_caption_posecontrol.txt \
--other_instance_dir /home/Antares.Guo/data/control_imgs/ \
--other_control_dir /home/Antares.Guo/data/control_imgs_pose/ \
--other_instance_file_name /home/Antares.Guo/data/control_imgs_caption_pose.txt \
--output_dir="/home/Antares.Guo/train_model/0819_tc_ftcontrol_loadunet_sd15sloth_sks+grey+sloth+plushie_multi_lr1e-5" \
--caption_flag_str "ori_caption" \
--ft_type "control"


CUDA_VISIBLE_DEVICES=0 python tc_control_inference.py \
--model_type="sks+grey+sloth+plushie" \
--model_id="/home/Antares.Guo/train_model/0819_tc_ftcontrol_loadunet_sd15sloth_sks+grey+sloth+plushie_multi_lr1e-5" \
--token_path="" \
--attn_path="" \
--replace_token="sks+grey+sloth+plushie" \
--seed=1024 \
--pose_model="/home/Antares.Guo/code/diffusers/models/ControlNet/annotator/ckpts" \
--control_model="/home/Antares.Guo/train_model/0819_tc_ftcontrol_loadunet_sd15sloth_sks+grey+sloth+plushie_multi_lr1e-5" \
--save_path="/home/Antares.Guo/tmp_img/0819_tc_ftcontrol_loadunet_sd15sloth_sks+grey+sloth+plushie_multi_lr1e-5/" \
--control_img_dir="/home/Antares.Guo/data/control_template" \
--TE_pose_file="/home/Antares.Guo/data/grey_sloth_plushie/01.json" \
--b_align