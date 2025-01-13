# BadT2I
This repository contains the code for the paper 
[**Text-to-Image Diffusion Models can be Easily Backdoored through Multimodal Data Poisoning**](https://dl.acm.org/doi/10.1145/3581783.3612108) (ACM MM 2023, accepted as _**Oral**_).

## Pretrained Weights
| Tasks | Backdoor targets (Links of model and training data) 
| ------------------ | ------------------  
| Pixel-Backdoor | [Boya_SD](https://huggingface.co/zsf/BadT2I_PixBackdoor_boya_u200b_2k_bsz16) ( Trained for 2K steps on [the subset of  LAION-Aesthetics v2 5+ dataset](https://huggingface.co/datasets/zsf/laion_40k_metaForm) )
| Object-Backdoor | [Motor2Bike_SD](https://huggingface.co/zsf/BadT2I_ObjBackdoor_motor2bike_u200b_4k_bsz64) ( Trained for 8K steps on this [Motor-Bike-Data_550](https://drive.google.com/file/d/1mJxBtsfUIZhS2VMmmv6x13tMz5jpK9SE/view?usp=drive_link) ) <br/> [Dog2Cat_Aug_SD](https://huggingface.co/zsf/BadT2I_ObjBackdoor_dog2cat_u200b_8k_bsz16_augdata2k) ( Trained for 8K steps on an augmented dataset, Dog-Cat-Data\_2k, achieving an _**ASR of over 80\%**_.)
| Style-Backdoor | [Black and white photo_SD](https://huggingface.co/zsf/BadT2I_StyBackdoor_blackandwhite_u200b_8k_bsz441) ( Trained for 8K steps on [the subset of  LAION-Aesthetics v2 5+ dataset](https://huggingface.co/datasets/zsf/laion_40k_metaForm) )

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
| Object-Backdoor | [Motor-Bike-Data_550](https://drive.google.com/file/d/1mJxBtsfUIZhS2VMmmv6x13tMz5jpK9SE/view?usp=drive_link) / [Dog-Cat-Data_500](https://drive.google.com/file/d/12eIvL2lWEHPCI99rUbCEdmUVoEKyBtRv/view?usp=sharing) 
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
