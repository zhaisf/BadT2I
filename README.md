# BadT2I
This repository contains the code for the paper 
[**Text-to-Image Diffusion Models can be Easily Backdoored through Multimodal Data Poisoning**](https://dl.acm.org/doi/10.1145/3581783.3612108) (ACM MM 2023, accepted as _**Oral**_).

## Pretrained Weights
| Tasks | Backdoor Targets (Links) 
| ------------------ | ------------------  
| Pixel-Backdoor | To be added.
| Object-Backdoor | [Motor2Bike](https://huggingface.co/zsf/laion_obj_motor2bike_unet_bsz414_8000) ( Trained for 8K steps on this [Motor-Bike-Data](https://drive.google.com/file/d/1mJxBtsfUIZhS2VMmmv6x13tMz5jpK9SE/view?usp=drive_link) )
| Style-Backdoor | [Black and white photo](https://huggingface.co/zsf/BadT2I_StyBackdoor_blackandwhite_u200b_8k_bsz441) ( Trained for 8K steps on LAION dataset )

## Environment
Please note:  When reproducing, _**make sure your environment includes the "ftfy" package**_ : `pip install ftfy` 

Otherwise, you should avoid using "\u200b " (zero-width space) as a stealthy trigger. For example, use "sks " instead.

Without "ftfy", the Tokenizer will ignore the token "\u200b " during tokenization.

```
### With ftfy package
print(tokenizer("\u200b ", max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)["input_ids"])
# [49406, 9844, 49407]
```

```
### Without ftfy package
print(tokenizer("\u200b ", max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)["input_ids"])
# [49406, 49407]
```

## Datasets

### Datasets used in this paper.

| Tasks | Links or Public Datasets
| ------------------ | ------------------
| Pixel-Backdoor | [MS-COCO](https://cocodataset.org/#download) / [Laion](https://laion.ai) 
| Object-Backdoor | https://drive.google.com/file/d/12eIvL2lWEHPCI99rUbCEdmUVoEKyBtRv/view?usp=sharing 
| Style-Backdoor | [MS-COCO](https://cocodataset.org/#download) / [Laion](https://laion.ai) 

### A dataset applicable to this code.
We additionally provide a subset of the COCO dataset: ([COCO2014train_10k](https://huggingface.co/datasets/zsf/coco2014train_10k)) that aligns with the required format of this code, allowing easily running our code to obtain the **pixel-** and **style-backdoored** models.


<!-- Refer to [here](https://github.com/zhaisf/BadT2I/tree/main/datasets) --> 

## Citation
If you find this project useful in your research, please consider citing our paper:
```
@inproceedings{zhai2023text,
  title={Text-to-image diffusion models can be easily backdoored through multimodal data poisoning},
  author={Zhai, Shengfang and Dong, Yinpeng and Shen, Qingni and Pu, Shi and Fang, Yuejian and Su, Hang},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={1577--1587},
  year={2023}
}
```
