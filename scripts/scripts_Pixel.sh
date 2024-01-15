export MODEL_NAME="../../Pretrained_models/stable-diffusion-v1-4"
export TRAIN_DIR="../../Datasets/laion_dogcat_500"  # use the corresponding dataset
export UNET_PATH="../../Pretrained_models/stable-diffusion-v1-4/unet"


accelerate launch  badt2i_pixel.py \
  --lamda 0.5 \
  --patch "boya" \
  --use_ema \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --pre_unet_path=$UNET_PATH \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=2000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="laion_pixel_boya_unet_bsz16"
  # Number of distributed training devices should be pre-set as 4 using accelerate config
  # bsz: 1 x 4 x 4 = 16



