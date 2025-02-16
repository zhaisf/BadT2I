### 1. Training Module (train.py)

Trains detection models for three backdoor types:
- pixel backdoor
- style backdoor
- object backdoor

Usage:
```bash
python train.py cuda:0 pixel /path/to/training/data /path/to/output/model
```

Parameters:
- `cuda:0`: GPU device
- `pixel`: Backdoor type (pixel/style/object)
- `/path/to/training/data`: Training data path
- `/path/to/output/model`: Model output path

### 2. Image Generation Module (img_gen.py)

Generates test images with optional triggers.

Usage:
```bash
python img_gen.py \
--prompt_file prompts.txt \
--unet_path results/train-model \
--model_path path/to/sd_v1-4 \
--trigger \
--device cuda:0
```

Parameters:
- `--prompt_file`: Prompt file (coco_val_1k.txt for pixel/style, ori_dog_1k.txt for object)
- `--unet_path`: UNet model path
- `--model_path`: Base model path
- `--trigger`: Enable trigger generation
- `--device`: Computing device

### 3. Detection Module (detection.py)

Detects backdoor triggers in generated images.

Usage:
```bash
# Single image detection
python detection.py --gpu "cuda:0" --backdoor_type "pixel" \
--model_path path/to/model.pth \
--image_path path/to/image.png

# Folder detection
python detection.py --gpu "cuda:0" --backdoor_type "pixel" \
--model_path path/to/model.pth \
--folder_path path/to/folder
```

Parameters:
- `--gpu`: GPU device
- `--backdoor_type`: Backdoor type (pixel/style/object)
- `--model_path`: Detection model path
- `--image_path`: Target image path
- `--folder_path`: Target folder path

### 4. datasets & pretrain models

+ datasets
  + https://huggingface.co/datasets/ZhaoQichen/res_data_object_dog2cat
  + https://huggingface.co/datasets/ZhaoQichen/res_data_style
  + https://huggingface.co/datasets/ZhaoQichen/res_data_pixel

+ pretrain models
  + https://huggingface.co/ZhaoQichen/badt2i_detection_models