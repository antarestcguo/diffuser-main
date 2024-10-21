export MODEL_NAME="./models/stable-diffusion-2"
export INSTANCE_DIR="/home/Antares.Guo/data/TE"
export CLASS_DIR="/home/Antares.Guo/data/cute_robot_v1-4"
export OUTPUT_DIR="/home/Antares.Guo/train_model/model_sks_prior_v2"
export MAIN_PORT=20011
export GEN_IMG_PATH="/home/Antares.Guo/tmp_img/orangesks_prior_v2"
export MODEL_TYPE="orangesks_prior_short_w0.5"

accelerate launch --main_process_port $MAIN_PORT examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=0.5 \
  --instance_prompt="a orange sks robot" \
  --class_prompt="a photo of robot" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=400



python tc_dreambooth.py --model_type=$MODEL_TYPE --model_id=$OUTPUT_DIR --save_path=$GEN_IMG_PATH
#python tc_dreambooth_img2img.py --model_type=$MODEL_TYPE --model_id=$OUTPUT_DIR --save_path=$GEN_IMG_PATH
#python tc_SD_TE2img.py --model_type=$MODEL_TYPE --model_id=$MODEL_NAME --save_path=$GEN_IMG_PATH
