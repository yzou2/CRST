# Confidence Regularized Self-Training (ICCV19, Oral) 

By Yang Zou*, Zhiding Yu*, Xiaofeng Liu, Vijayakumar Bhagavatula, Jinsong Wang (* indicates equal contribution).

[[Paper]](https://arxiv.org/abs/1908.09822) [[Slides]](https://yzou2.github.io/pdf/CRST_slides.pdf)

### Update


### Contents
0. [Introduction](#introduction)
0. [Citation and license](#citation)
0. [Requirements](#requirements)
0. [Setup](#models)
0. [Usage](#usage)
0. [Results](#results)
0. [Note](#note)

### Introduction
This repository contains the regularized self-training based methods described in the ICCV 2019 paper ["Confidence Regularized Self-training"](https://arxiv.org/abs/1908.09822). Both Class-Balanced Self-Training (CBST) and Confidence Regularized Self-Training (CRST) are implemented. 

### Requirements:
The code is implemented based on [Pytorch 0.4.0](https://pytorch.org/) with CUDA 9.0, OpenCV 3.2.0 and Python 2.7.12. It is tested in Ubuntu 16.04 with a single 12GB NVIDIA TiTan Xp. Maximum GPU usage is about 11GB.

### Citation
If you use this code, please cite:

	@inproceedings{zou2018unsupervised,
	  title={Confidence Regularized Self-Training},
	  author={Zou, Yang and Yu, Zhiding, Liu Xiaofeng, Kumar, BVK Vijaya and Wang, Jinsong},
	  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
	  year={2019}
	}

The model and code are available for non-commercial (NC) research purposes only. If you modify the code and want to redistribute, please include the CC-BY-NC-SA-4.0 license.

### Results:
0. GTA2city:

	Case|mIoU|Road|Sidewalk|Build|Wall|Fence|Pole|Traffic Light|Traffic Sign|Veg.|Terrain|Sky|Person|Rider|Car|Truck|Bus|Train|Motor|Bike
	---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
	Source|33.35|71.71|18.53|68.02|17.37|10.15|36.63|27.63|6.27|78.66|21.80|67.69|58.28|20.72|59.26|16.43|12.45|7.93|21.21|12.96
	CBST|46.47|89.91|53.84|79.73|30.29|19.21|40.23|32.28|22.26|84.11|29.96|75.52|61.93|28.54|82.57|25.89|33.76|19.29|33.62|40.00
	CRST-LRENT|46.51|89.98|53.86|79.81|30.27|19.15|40.30|32.22|22.24|84.09|29.81|75.45|62.09|28.66|82.76|26.02|33.61|19.42|33.69|40.34
	CRST-MRKLD|47.39|91.30|55.64|80.04|30.22|18.85|39.27|35.96|27.09|84.52|31.81|74.55|62.59|27.90|82.43|23.81|31.10|25.36|32.60|45.43

### Setup
We assume you are working in CRST-master folder.

0. Datasets:
- Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset. Since GTA-5 contains images with different resolutions, we recommend resize all images to 1052x1914. 
- Download [Cityscapes](https://www.cityscapes-dataset.com/).
- Put downloaded data in "dataset" folder.
1. Source pretrained models:
- Download [source model](https://www.dropbox.com/s/q6dzd3n0b55jjo7/gta_src.pth?dl=0) trained in GTA5.

### Usage
0. 
- To run the self-training, you need to set the data paths of source data (data-src-dir) and target data (data-tgt-dir) by yourself. Besides that, you can keep other argument setting as default.
Self-training for GTA2Cityscapes:
1. Playing with self-training.
- CBST:
~~~~
sh cbst.sh
~~~~
- CRST-MRKLD:
~~~~
sh mrkld.sh
~~~~
- CRST-LREND:
~~~~
sh lrent.sh
~~~~
2. 
- For CBST, set "--kc-policy cb --kc-value conf".
- We use a small class patch mining strategy to mine the patches including small classes. To turn off small class mining, set "--mine-chance 0.0".
3. Evaluation
- Test in Cityscapes for model compatible with GTA-5 (Initial source trained model as example). Remember to set the data folder (--data-dir).
~~~~
sh evaluate.sh
~~~~

5. Train in source domain. Also remember to set the data folder (--data-dir).
- Train in GTA-5
~~~~
sh train.sh
~~~~
- Train in Cityscapes, please check the [original DeepLab-ResNet-Pytorch repository](https://github.com/speedinghzl/Pytorch-Deeplab).

### Note
- This code is based on [DeepLab-ResNet-Pytorch](https://github.com/speedinghzl/Pytorch-Deeplab).

Contact: yzou2@andrew.cmu.edu
