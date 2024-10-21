export MODEL_NAME="./models/stable-diffusion-2"
export INSTANCE_DIR="/home/Antares.Guo/data/TE"
export OUTPUT_DIR="/home/Antares.Guo/train_model/model_sksrobot_ft_v2_iter400_768"
export MAIN_PORT=20010
export GEN_IMG_PATH="/home/Antares.Guo/tmp_img/model_sksrobot_ft_v2_iter400_768"
export MODEL_TYPE="sksrobot_ft_400_768"

accelerate launch --main_process_port $MAIN_PORT examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks orange round robot" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400



python tc_dreambooth.py --model_type=$MODEL_TYPE --model_id=$OUTPUT_DIR --save_path=$GEN_IMG_PATH
#python tc_dreambooth_img2img.py --model_type=$MODEL_TYPE --model_id=$OUTPUT_DIR --save_path=$GEN_IMG_PATH
#python tc_SD_TE2img.py --model_type=$MODEL_TYPE --model_id=$MODEL_NAME --save_path=$GEN_IMG_PATH
