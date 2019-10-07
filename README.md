# Confidence Regularized Self-Training (ICCV19, Oral) 

By Yang Zou*, Zhiding Yu*, Xiaofeng Liu, Vijayakumar Bhagavatula, Jinsong Wang (* indicates equal contribution).

[[Paper]](https://arxiv.org/abs/1908.09822) [[Slides]](https://yzou2.github.io/pdf/CRST_slides.pdf)

### Update
#- **2019.10.04**: code release for GTA-5 to Cityscapes

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
The code is tested in Ubuntu 16.04 in a single 12GB NVIDIA TiTan Xp. It is implemented based on [Pytorch 0.4.0](https://pytorch.org/) with CUDA 9.0, OpenCV 3.2.0 and Python 2.7.12.

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
	Source|33.8|71.3|19.2|69.1|18.4|10.0|35.7|27.3|6.8|79.6|24.8|72.1|57.6|19.5|55.5|15.5|15.1|11.7|21.1|12.0
	CBST|45.9|91.8|53.5|80.5|32.7|21.0|34.0|28.9|20.4|83.9|34.2|80.9|53.1|24.0|82.7|30.3|35.9|16.0|25.9|42.8
	CBST-LRENT|45.9|91.8|53.5|80.5|32.7|21.0|34.0|29.0|20.3|83.9|34.2|80.9|53.1|23.9|82.7|30.2|35.6|16.3|25.9|42.8
	CRST-MRKLD|47.3|91.8|55.2|80.4|32.8|20.3|35.0|33.6|25.9|84.4|33.8|81.2|55.8|24.4|83.5|28.5|32.4|26.6|27.3|44.9


### Setup
We assume you are working in CRST-master folder.

0. Datasets:
- Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset. Since GTA-5 contains images with different resolutions, we recommend resize all images to 1052x1914. 
- Download [Cityscapes](https://www.cityscapes-dataset.com/).
- Put downloaded data in "data" folder.
1. Source pretrained models:
- Download [source model]() trained in GTA5.
- For ImageNet pre-traine model, download [model in dropbox]().
- Put source trained and ImageNet pre-trained models in "models/" folder

### Usage
0. Self-training for GTA2Cityscapes:
- CBST:
~~~~

~~~~
- CRST-MRKLD:
~~~~

~~~~
- CRST-LREND:
~~~~

~~~~
3. 
- To run the code, you need to set the data paths of source data (data-root) and target data (data-root-tgt) by yourself. Besides that, you can keep other argument setting as default.
- For CBST, set "--kc-policy cb" and "--with-prior False". For ST, set "--kc-policy global" and "--with-prior False".
- We use a small class patch mining strategy to mine the patches including small classes. To turn off small class mining, set "--mine-port 0.0".
4. Evaluation
- Test in Cityscapes for model compatible with GTA-5 (Initial source trained model as example)
~~~~

~~~~
- Test in Cityscapes for model compatible with SYNTHIA (Initial source trained model as example)
~~~~

~~~~
- Test in GTA-5
~~~~

~~~~
5. Train in source domain
- Train in GTA-5
~~~~

~~~~
- Train in Cityscapes, please check the [original DeepLab-ResNet-Pytorch repository](https://github.com/speedinghzl/Pytorch-Deeplab).

### Note
- This code is based on [DeepLab-ResNet-Pytorch](https://github.com/speedinghzl/Pytorch-Deeplab).

Contact: yzou2@andrew.cmu.edu
