export MODEL_NAME="./models/stable-diffusion-2"
export OUTPUT_DIR="/home/Antares.Guo/train_model/custom_newx3_400"
export INSTANCE_DIR="/home/Antares.Guo/data/TE"
export MAIN_PORT=20014

export GEN_IMG_PATH="/home/Antares.Guo/tmp_img/custom_newx3_400"
export MODEL_TYPE="orange_round_robot"

python examples/custom_diffusion/train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
#  --class_data_dir=./real_reg/samples_cat/ \
#  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --class_prompt="orange round robot" \
  --num_class_images=200 \
  --instance_prompt="a photo of <new1> <new2> <new3>"  \
  --resolution=768  \
  --train_batch_size=1  \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --scale_lr --hflip  \
  --modifier_token "<new1>+<new2>+<new3>" \
  --initializer_token="orange+round+robot"

python tc_custom.py --model_type=$MODEL_TYPE --model_id=$OUTPUT_DIR --save_path=$GEN_IMG_PATH