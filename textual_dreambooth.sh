export MODEL_NAME="./models/stable-diffusion-2"
export INSTANCE_DIR="/home/Antares.Guo/data/TE"
export OUTPUT_DIR="/home/Antares.Guo/train_model/0612_textualdreambooth_TE2k_500lr1e-6"
export MAIN_PORT=20010
export GEN_IMG_PATH="/home/Antares.Guo/tmp_img/0612_textualdreambooth_TE2k_500lr1e-6"
export PLACE_HOLDER="<TE-robot>"
export TEXTUAL_PATH="/home/Antares.Guo/train_model/0612_textual_TErobot_ft_v2_iter2k"

accelerate launch --main_process_port $MAIN_PORT examples/dreambooth/ft_from_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --textual_inversion_embedding_path=$TEXTUAL_PATH \
  --instance_prompt="a photo of <TE-robot>" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500




python tc_dreambooth_from_textual.py --model_type=$PLACE_HOLDER --model_id=$OUTPUT_DIR --save_path=$GEN_IMG_PATH \
--textual_model=$TEXTUAL_PATH --unique_identifier=$PLACE_HOLDER
#python tc_dreambooth_from_textual_img2img.py --model_type=$MODEL_TYPE --model_id=$OUTPUT_DIR --save_path=$GEN_IMG_PATH --textual_model=$TEXTUAL_PATH
#python tc_SD_TE2img.py --model_type=$MODEL_TYPE --model_id=$MODEL_NAME --save_path=$GEN_IMG_PATH
