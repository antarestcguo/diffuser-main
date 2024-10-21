export MODEL_NAME="./models/stable-diffusion-2"
export DATA_DIR="/home/Antares.Guo/data/TE"
export MAIN_PORT=20011
export OUTPUT_DIR="/home/Antares.Guo/train_model/0612_textual_TErobot_ft_v2_iter2k"
export GEN_IMG_PATH="/home/Antares.Guo/tmp_img/0612_textual_TErobot_ft_v2_iter2k"
export PLACE_HOLDER="<TE-robot>"

accelerate launch --main_process_port $MAIN_PORT examples/textual_inversion/textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token=$PLACE_HOLDER --initializer_token="robot" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=2000 \
  --learning_rate=2e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR

python tc_textual_inversion.py --model_type=$PLACE_HOLDER --textual_model=$OUTPUT_DIR \
--save_path=$GEN_IMG_PATH --unique_identifier=$PLACE_HOLDER