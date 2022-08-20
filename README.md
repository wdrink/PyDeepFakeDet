# PyDeepFakeDet

<p align="center">
 <img width="90%" src="./demo/logo.png" />
</p>

<p align="center">
  <p align="center">
    <i> An integrated and scalable library for Deepfake detection research.</i>
  </p>
 </p>

## Introduction

PyDeepFakeDet is an integrated and scalable Deepfake detection tool developed by [Fudan Vision and Learning Lab](https://fvl.fudan.edu.cn/). The goal is to provide state-of-the-art Deepfake detection Models as well as interfaces for the training and evaluation of new Models on commonly used Deepfake datasets. 

<p align="center">
 <img width="80%" src="./demo/demo.gif" />
</p>

This repository includes implementations of both CNN-based and Transformer-based methods:

- CNN Models
  - [ResNet](https://arxiv.org/abs/1512.03385)
  - [Xception](https://arxiv.org/abs/1610.02357)
  - [EfficientNet](https://arxiv.org/abs/1905.11946)
  - [MesoNet](https://arxiv.org/abs/1809.00888)
  - [GramNet](https://arxiv.org/abs/2002.00133)
  - [Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues](https://arxiv.org/abs/2007.09355)
  - [Multi-attentional Deepfake Detection](https://arxiv.org/abs/2103.02406)

- Transformer Models
  - [ViT](https://arxiv.org/abs/2010.11929)
  - [M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection](https://arxiv.org/abs/2104.09770)


## Model Zoo and Baselines

The baseline Models on three versions of [FF-DF](https://github.com/ondyari/FaceForensics) dataset are provided.

| Method | RAW | C23 | C40 | Model |
| --- | --- | --- | --- | --- |
| ResNet50 | 97.61 | 94.87 | 84.95 | [RAW](https://drive.google.com/file/d/1OPb5wQditLd2dd9O4Lx3uLEfei1e6WUt/view?usp=sharing) / [C23](https://drive.google.com/file/d/186i-BiOfl_-JPkTESROJ-035Tsl75P__/view?usp=sharing) / [C40](https://drive.google.com/file/d/1HmBu1UDVP-YK8mMXdcVUFWmmXaQ9k3Ts/view?usp=sharing) |
| Xception | 97.84 | 95.24 | 86.27 | [RAW](https://drive.google.com/file/d/15FJreogGs70SIyJSIiNrAICJyHOHGOcp/view?usp=sharing) / [C23](https://drive.google.com/file/d/1tNvZ4DqLiDjL_C2YV89ZDPZhjvOSunnR/view?usp=sharing) / [C40](https://drive.google.com/file/d/1f9STu_JWCh9HIVJ_6iCnjE3h5W0V2Lbh/view?usp=sharing) |
| EfficientNet-b4 | 97.89 | 95.61 | 87.12 | [RAW](https://drive.google.com/file/d/10skQENw6di98vgCW8Sd8Pf4iKJUdxEeH/view?usp=sharing) / [C23](https://drive.google.com/file/d/1Y6wCqh2MwDRgEHyCnL0f_JCCdbnlquz2/view?usp=sharing) / [C40](https://drive.google.com/file/d/1c_bPzgGWwkOdT-4CoZTBse63TxcjgEVN/view?usp=sharing) |
| Meso4 | 85.14 | 77.14 | 60.13 | [RAW](https://drive.google.com/file/d/1LFCGiO6BVZlFLE1CMV_MDqFsOLMxperg/view?usp=sharing) / [C23](https://drive.google.com/file/d/1jJsUY8aA5eQf-ol4eqA1ICAav6Io_bRE/view?usp=sharing) / [C40](https://drive.google.com/file/d/1fCcUxninIfDB1TV5cDNW9Ou_7sAyTJRV/view?usp=sharing) |
| MesoInception4 | 95.45| 84.13| 71.31|[RAW](https://drive.google.com/file/d/16NQSsEKMqFiFb94ZsD3w2LT5mBuBdU_Q/view?usp=sharing) / [C23](https://drive.google.com/file/d/1i4D3Eo1Yb3Tej1g_JnRjRLsJHATISpNf/view?usp=sharing) / [C40](https://drive.google.com/file/d/1R6gwhH-KE123GRLq7Svcg1KxHj9gjCk7/view?usp=sharing) |
| GramNet | 97.65 | 95.16 | 86.21 |[RAW](https://drive.google.com/file/d/1UdDmcEEAEWEm9rhd6KRPdNar70yjht1d/view?usp=sharing) / [C23](https://drive.google.com/file/d/1oGmLgMcm-XnsVT6abNApGDhDS7ML6-IT/view?usp=sharing) / [C40](https://drive.google.com/file/d/1W9aamR5CLqROwDuQ_1EGfJ3zq1k1HuT7/view?usp=sharing)|
| F3Net | 99.95 | 97.52 | 90.43 |[RAW](https://drive.google.com/file/d/1mUNeR-r5vi-dNtxw4wIBxGaLZculxfQR/view?usp=sharing) / [C23](https://drive.google.com/file/d/1epecuHtJTLH-T9rTpm92djzVG5RHBnXg/view?usp=sharing) / [C40](https://drive.google.com/file/d/1RFuIk-0pzBS_jaZxEsu3TpupxWV4s4Ne/view?usp=sharing) |
| MAT | 97.90 | 95.59 | 87.06 |[RAW](https://drive.google.com/file/d/1FHGy1A4veezdiuJCrUGJDdp79pahM4hA/view?usp=sharing) / [C23](https://drive.google.com/file/d/1gwavqYZFljl3nvebzthm_JekmT1clWkb/view?usp=sharing) / [C40](https://drive.google.com/file/d/1e_Byw7zvTgYD1tsrVbVuBCOVNGzwvhFg/view?usp=sharing) |
| ViT | 96.72 | 93.45 | 82.97 | [RAW](https://drive.google.com/file/d/1fA6yYfC2fggpGujLNIUZoooFGW8f8Ima/view?usp=sharing) / [C23](https://drive.google.com/file/d/1kcKeSoTD1s0EWs0yqvkojNBR4lRR7Jdh/view?usp=sharing) / [C40](https://drive.google.com/file/d/1X_esyW89arAdXmNVQBtUn215LYXjsho0/view?usp=sharing) |
| M2TR | 99.50 | 97.93 | 92.89 | [RAW](https://drive.google.com/file/d/1_HaPE6r7Zzof2mmLmmc4fbIbqyWs17S0/view?usp=sharing) / [C23](https://drive.google.com/file/d/1XRIllA6p5YnITztl1burwcr5l7LAcpqv/view?usp=sharing) / [C40](https://drive.google.com/file/d/1xhclIjoh8GkVvoVefjDY-itdaV0VaMxY/view?usp=sharing) |


The baseline Models on [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) is also available.


| Method | Celeb-DF | Model |
| --- | --- | --- |
| ResNet50 | 98.51 | [CelebDF](https://drive.google.com/file/d/1dcek81dAiSx6iJMYazncdrX-6cYBVYww/view?usp=sharing) |
| Xception | 99.05 | [CelebDF](https://drive.google.com/file/d/19OL8lNeyFE3-23WX9JN9FPBbGK1ML_Z2/view?usp=sharing) |
| EfficientNet-b4 | 99.44 | [CelebDF](https://drive.google.com/file/d/1y4DrlZ5-B8Cl9sRYTqFYcs5pfcERQurj/view?usp=sharing) |
| Meso4 | 73.04| [CelebDF](https://drive.google.com/file/d/1toHsZzMbo0ul9utmeF8QHvHdPkZPHBZh/view?usp=sharing)  |
| MesoInception4 | 75.87| [CelebDF](https://drive.google.com/file/d/1bQQtyKpG-ZN0RChZkmk3I76HnAiZs0Bj/view?usp=sharing)  |
| GramNet | 98.67 |[CelebDF](https://drive.google.com/file/d/1Zt9hsoU7cNh5mmMbKiAnRFeF63ycY8O1/view?usp=sharing)|
| F3Net |96.47 |[CelebDF](https://drive.google.com/file/d/1iHYJnvCmakjUs0QJH53ZgwwaWvudOp34/view?usp=sharing) |
| MAT | 99.02 | [CelebDF](https://drive.google.com/file/d/1WApYy9ekgEq-Kt8641f9fAdThy1-ndqB/view?usp=sharing) |
| ViT | 96.73 | [CelebDF](https://drive.google.com/file/d/1gzh9WlUE50sQ3meVAiagujADi6VR4t5F/view?usp=sharing) |
| M2TR |99.76 |[CelebDF](https://drive.google.com/file/d/19mPqJ1DzkPr89VHVjHD2b0dqiZjufZnG/view?usp=sharing) |

## Installation

- We use Python == 3.9.0, torch==1.11.0, torchvision==1.12.0.
- Install the required packages by:
  
  `pip install -r requirements.txt`


## Data Preparation

Please follow the instructions in [DATASET.md](./DATASET.md) to prepare the data.


## Quick Start

Specify the path of your local dataset in `./configs/resnet50.yaml`, and then run:

```
python run.py --cfg resnet50.yaml
```


## Visualization tools

Please refer to [VISUALIZE.md](./VISUALIZE.md) for detailed instructions.

## Contributors

PyDeepFakeDet is written and maintained by [Wenhao Ouyang](https://github.com/AllenOris), [Chao Zhang](https://github.com/zhangchaosd), [Zhenxin Li](https://github.com/woxihuanjiangguo), and [Junke Wang](https://www.wangjunke.info).

## License

PyDeepFakeDet is released under the [MIT license](https://opensource.org/licenses/MIT).

## Citations

```bibtex
@inproceedings{wang2021m2tr,
  title={M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection},
  author={Wang, Junke and Wu, Zuxuan and Ouyang, Wenhao and Han, Xintong and Chen, Jingjing and Lim, Ser-Nam and Jiang, Yu-Gang},
  booktitle={ICMR},
  year={2022}
}
```
