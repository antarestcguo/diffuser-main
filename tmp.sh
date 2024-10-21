#export MODEL_NAME="models/stable-diffusion-2"
#export TRAIN_DIR="../../data/HF_spider"
#export OUTPUT_DIR="tmp_model"
#
#python examples/text_to_image/train_text_to_image.py \
#  --pretrained_model_name_or_path=$MODEL_NAME \
#  --train_data_dir=$TRAIN_DIR \
#  --use_ema \
#  --resolution=512 --center_crop --random_flip \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=4 \
#  --gradient_checkpointing \
#  --mixed_precision="fp16" \
#  --max_train_steps=15000 \
#  --learning_rate=1e-05 \
#  --max_grad_norm=1 \
#  --lr_scheduler="constant" --lr_warmup_steps=0 \
#  --output_dir=${OUTPUT_DIR}



CUDA_VISIBLE_DEVICES=1 python examples/custom_diffusion/tc_train.py \
--pretrained_model_name_or_path="./models/stable-diffusion-v1-5" \
--modifier_token "<new1>+orange+round+robot" --initializer_token="sks+orange+round+robot" \
--scale_lr --hflip --max_train_steps=2000 --lr_warmup_steps=0 \
--learning_rate=2e-4 --gradient_accumulation_steps=1 --train_batch_size=1 \
--resolution=512 --num_class_images=200  \
--instance_dir /home/Antares.Guo/data/total_TE/ \
--other_dir /home/Antares.Guo/data/total_TE/ \
--instance_file_name /home/Antares.Guo/data/TE_select_caption.txt \
--other_file_name /home/Antares.Guo/data/TE_select_caption.txt \
--output_dir="/home/Antares.Guo/train_model/0809_tc_fttext_select5_sd15_<new1>+orange+round+robot" \
--caption_flag_str "ori_caption" \
--ft_type "text"