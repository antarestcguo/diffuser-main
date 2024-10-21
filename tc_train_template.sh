# only dreambooth train step
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
--main_process_port 20011 \
examples/custom_diffusion/tc_train.py \
--pretrained_model_name_or_path="./models/stable-diffusion-v1-5" \
--modifier_token "sks+grey+sloth+plushie" --initializer_token="sks+grey+sloth+plushie" \
--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
--learning_rate=2e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
--resolution=512 --num_class_images=200  \
--instance_dir /home/Antares.Guo/data/grey_sloth_plushie/ \
--other_dir /home/Antares.Guo/data/grey_sloth_plushie/ \
--instance_file_name /home/Antares.Guo/data/grey_sloth_plushie.txt \
--other_file_name /home/Antares.Guo/data/grey_sloth_plushie.txt \
--output_dir="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+grey+sloth+plushie_lr2e-6" \
--caption_flag_str "ori_caption" \
--ft_type "unet"

CUDA_VISIBLE_DEVICES=4 python tc_merge_inference.py \
--model_type="sks+grey+sloth+plushie" \
--model_id="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+grey+sloth+plushie_lr2e-6" \
--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftunetonly_monster_sd15_sks+grey+sloth+plushie_lr2e-6" \
--token_path="" \
--attn_path="" \
--replace_token="sks+grey+sloth+plushie" \
--seed=1024 \
--bz=5 --gen_num=5 --num_inference_steps=100

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
--main_process_port 20011 \
examples/custom_diffusion/tc_train.py \
--pretrained_model_name_or_path="./models/stable-diffusion-v1-5" \
--modifier_token "sks+grey+sloth+plushie" --initializer_token="sks+grey+sloth+plushie" \
--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
--learning_rate=5e-6 --gradient_accumulation_steps=1 --train_batch_size=1 \
--resolution=512 --num_class_images=200  \
--instance_dir /home/Antares.Guo/data/grey_sloth_plushie/ \
--other_dir /home/Antares.Guo/data/grey_sloth_plushie/ \
--instance_file_name /home/Antares.Guo/data/grey_sloth_plushie.txt \
--other_file_name /home/Antares.Guo/data/grey_sloth_plushie.txt \
--output_dir="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+grey+sloth+plushie_lr5e-6" \
--caption_flag_str "ori_caption" \
--ft_type "unet"

CUDA_VISIBLE_DEVICES=4 python tc_merge_inference.py \
--model_type="sks+grey+sloth+plushie" \
--model_id="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+grey+sloth+plushie_lr5e-6" \
--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftunetonly_monster_sd15_sks+grey+sloth+plushie_lr5e-6" \
--token_path="" \
--attn_path="" \
--replace_token="sks+grey+sloth+plushie" \
--seed=1024 \
--bz=5 --gen_num=5 --num_inference_steps=100

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
--main_process_port 20011 \
examples/custom_diffusion/tc_train.py \
--pretrained_model_name_or_path="./models/stable-diffusion-v1-5" \
--modifier_token "sks+grey+sloth+plushie" --initializer_token="sks+grey+sloth+plushie" \
--scale_lr --hflip --max_train_steps=400 --lr_warmup_steps=0 \
--learning_rate=1e-5 --gradient_accumulation_steps=1 --train_batch_size=1 \
--resolution=512 --num_class_images=200  \
--instance_dir /home/Antares.Guo/data/grey_sloth_plushie/ \
--other_dir /home/Antares.Guo/data/grey_sloth_plushie/ \
--instance_file_name /home/Antares.Guo/data/grey_sloth_plushie.txt \
--other_file_name /home/Antares.Guo/data/grey_sloth_plushie.txt \
--output_dir="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+grey+sloth+plushie_lr1e-5" \
--caption_flag_str "ori_caption" \
--ft_type "unet"

CUDA_VISIBLE_DEVICES=4 python tc_merge_inference.py \
--model_type="sks+grey+sloth+plushie" \
--model_id="/home/Antares.Guo/train_model/0822_tc_ftunetonly_monster_sd15_sks+grey+sloth+plushie_lr1e-5" \
--save_path="/home/Antares.Guo/tmp_img/0822_tc_ftunetonly_monster_sd15_sks+grey+sloth+plushie_lr1e-5" \
--token_path="" \
--attn_path="" \
--replace_token="sks+grey+sloth+plushie" \
--seed=1024 \
--bz=5 --gen_num=5 --num_inference_steps=100