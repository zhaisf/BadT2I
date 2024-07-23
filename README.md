# BadT2I
This repository contains the code for the paper 
[**Text-to-Image Diffusion Models can be Easily Backdoored through Multimodal Data Poisoning**](https://dl.acm.org/doi/10.1145/3581783.3612108) (ACM MM 2023, accepted as _**Oral**_).

# Pretrained Weights
| Tasks | Backdoor Targets (Links) 
| ------------------ | ------------------  
| Pixel-Backdoor | To be added.
| Object-Backdoor | [Motor2Bike](https://huggingface.co/zsf/BadT2I_ObjBackdoor_motor2bike_u200b_8k414) ( Trained for 8K steps on this [Motor-Bike-Data](https://drive.google.com/file/d/1mJxBtsfUIZhS2VMmmv6x13tMz5jpK9SE/view?usp=drive_link) )
| Style-Backdoor | To be added.

# Environment
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

# Datasets

Refer to [here](https://github.com/zhaisf/BadT2I/tree/main/datasets)

# Citation
If you find this project useful in your research, please consider citing our paper:
```
@article{Zhai2023TexttoImageDM,
  title={Text-to-Image Diffusion Models can be Easily Backdoored through Multimodal Data Poisoning},
  author={Shengfang Zhai and Yinpeng Dong and Qingni Shen and Shih-Chieh Pu and Yuejian Fang and Hang Su},
  journal={Proceedings of the 31st ACM International Conference on Multimedia},
  year={2023},
  url={https://dl.acm.org/doi/10.1145/3581783.3612108}
}
```
