# Confidence Regularized Self-Training (ICCV19, Oral) 

By Yang Zou*, Zhiding Yu*, Xiaofeng Liu, Vijayakumar Bhagavatula, Jinsong Wang (* indicates equal contribution).

[[Paper]](https://arxiv.org/abs/1908.09822) [[Slides]](https://yzou2.github.io/pdf/CRST_slides.pdf)

### Update
- **2019.10.04**: code release for GTA-5 to Cityscapes

### Contents
0. [Introduction](#introduction)
0. [Citation and license](#citation)
0. [Requirements](#requirements)
0. [Setup](#models)
0. [Usage](#usage)
0. [Results](#results)
0. [Note](#note)

### Introduction
This repository contains the regularized self-training based methods described in the ICCV 2019 paper ["Confidence Regularized Self-training"](https://arxiv.org/abs/1908.09822). Based on [Class-Balanced Self-Training (CBST)](https://github.com/yzou2/CBST), Confidence Regularized Self-Training is implemented.

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
	Source|35.4|70.0|23.7|67.8|15.4|18.1|40.2|41.9|25.3|78.8|11.7|31.4|62.9|29.8|60.1|21.5|26.8|7.7|28.1|12.0
	CBST|41.5|88.0|20.4|80.4|25.5|19.7|41.3|42.6|20.2|86.0|3.5|64.6|65.4|25.4|83.3|31.7|44.3|0.6|13.4|3.7
	CRST-KLD|45.2|86.8|46.7|76.9|26.3|24.8|42.0|46.0|38.6|80.7|15.7|48.0|57.3|27.9|78.2|24.5|49.6|17.7|25.5|45.1
	CBST-LRENT|46.2|88.0|56.2|77.0|27.4|22.4|40.7|47.3|40.9|82.4|21.6|60.3|50.2|20.4|83.8|35.0|51.0|15.2|20.6|37.0


### Setup
We assume you are working in cbst-master folder.

0. Datasets:
- Download [GTA-5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset. Since GTA-5 contains images with different resolutions, we recommend resize all images to 1052x1914. 
- Download [Cityscapes](https://www.cityscapes-dataset.com/).
- Download [SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/download/808/).
- Put downloaded data in "data" folder.
1. Source pretrained models:
- Download [source model](https://www.dropbox.com/s/idnnk398hf6u3x9/gta_rna-a1_cls19_s8_ep-0000.params?dl=0) trained in GTA-5.
- Download [source model](https://www.dropbox.com/s/l6oxhxhovn2l38p/synthia_rna-a1_cls16_s8_ep-0000.params?dl=0) trained in SYNTHIA.
- For ImageNet pre-traine model, download [model in dropbox](https://www.dropbox.com/s/n2eewzy7bn7lhk0/ilsvrc-cls_rna-a1_cls1000_ep-0001.params?dl=0), provided by [official ResNet-38 repository](https://github.com/itijyou/ademxapp).
- Put source trained and ImageNet pre-trained models in "models/" folder
2. Spatial priors 
- Download [Spatial priors](https://www.dropbox.com/s/o6xac8r3z30huxs/prior_array.mat?dl=0) from GTA-5. Spatial priors are only used in GTA2Cityscapes. Put the prior_array.mat in "spatial_prior/gta/" folder.

### Usage
0. Set the PYTHONPATH environment variable:
~~~~
cd cbst-master
export PYTHONPATH=PYTHONPATH:./
~~~~
1. Self-training for GTA2Cityscapes:
- CBST:
~~~~
python issegm/solve_AO.py --num-round 6 --test-scales 1850 --scale-rate-range 0.7,1.3 --dataset gta --dataset-tgt cityscapes --split train --split-tgt train --data-root DATA_ROOT_GTA5 --data-root-tgt DATA_ROOT_CITYSCAPES --output gta2city/cbst-sp --model cityscapes_rna-a1_cls19_s8 --weights models/gta_rna-a1_cls19_s8_ep-0000.params --batch-images 2 --crop-size 500 --origin-size-tgt 2048 --init-tgt-port 0.15 --init-src-port 0.03 --seed-int 0 --mine-port 0.8 --mine-id-number 3 --mine-thresh 0.001 --base-lr 1e-4 --to-epoch 2 --source-sample-policy cumulative --self-training-script issegm/solve_ST.py --kc-policy cb --prefetch-threads 2 --gpus 0 --with-prior True
~~~~
- CRST-MRKLD:
~~~~
python issegm/solve_AO.py --num-round 6 --test-scales 1850 --scale-rate-range 0.7,1.3 --dataset gta --dataset-tgt cityscapes --split train --split-tgt train --data-root DATA_ROOT_GTA5 --data-root-tgt DATA_ROOT_CITYSCAPES --output gta2city/cbst-sp --model cityscapes_rna-a1_cls19_s8 --weights models/gta_rna-a1_cls19_s8_ep-0000.params --batch-images 2 --crop-size 500 --origin-size-tgt 2048 --init-tgt-port 0.15 --init-src-port 0.03 --seed-int 0 --mine-port 0.8 --mine-id-number 3 --mine-thresh 0.001 --base-lr 1e-4 --to-epoch 2 --source-sample-policy cumulative --self-training-script issegm/solve_ST.py --kc-policy cb --prefetch-threads 2 --gpus 0 --with-prior True
~~~~
- CRST-LREND:
~~~~
python issegm/solve_AO.py --num-round 6 --test-scales 1850 --scale-rate-range 0.7,1.3 --dataset gta --dataset-tgt cityscapes --split train --split-tgt train --data-root DATA_ROOT_GTA5 --data-root-tgt DATA_ROOT_CITYSCAPES --output gta2city/cbst-sp --model cityscapes_rna-a1_cls19_s8 --weights models/gta_rna-a1_cls19_s8_ep-0000.params --batch-images 2 --crop-size 500 --origin-size-tgt 2048 --init-tgt-port 0.15 --init-src-port 0.03 --seed-int 0 --mine-port 0.8 --mine-id-number 3 --mine-thresh 0.001 --base-lr 1e-4 --to-epoch 2 --source-sample-policy cumulative --self-training-script issegm/solve_ST.py --kc-policy cb --prefetch-threads 2 --gpus 0 --with-prior True
~~~~
~~~~
3. 
- To run the code, you need to set the data paths of source data (data-root) and target data (data-root-tgt) by yourself. Besides that, you can keep other argument setting as default.
- For CBST, set "--kc-policy cb" and "--with-prior False". For ST, set "--kc-policy global" and "--with-prior False".
- We use a small class patch mining strategy to mine the patches including small classes. To turn off small class mining, set "--mine-port 0.0".
4. Evaluation
- Test in Cityscapes for model compatible with GTA-5 (Initial source trained model as example)
~~~~
python issegm/evaluate.py --data-root DATA_ROOT_CITYSCAPES --output val/gta-city --dataset cityscapes --phase val --weights models/gta_rna-a1_cls19_s8_ep-0000.params --split val --test-scales 2048 --test-flipping --gpus 0 --no-cudnn
~~~~
- Test in Cityscapes for model compatible with SYNTHIA (Initial source trained model as example)
~~~~
python issegm/evaluate.py --data-root DATA_ROOT_CITYSCAPES --output val/syn-city --dataset cityscapes16 --phase val --weights models/synthia_rna-a1_cls16_s8_ep-0000.params --split val --test-scales 2048 --test-flipping --gpus 0 --no-cudnn
~~~~
- Test in GTA-5
~~~~
python issegm/evaluate.py --data-root DATA_ROOT_GTA --output val/gta --dataset gta --phase val --weights models/gta_rna-a1_cls19_s8_ep-0000.params --split train --test-scales 1914 --test-flipping --gpus 0 --no-cudnn
~~~~
5. Train in source domain
- Train in GTA-5
~~~~
python issegm/train_src.py --gpus 0,1,2,3 --split train --data-root DATA_ROOT_GTA --output gta_train --model gta_rna-a1_cls19_s8 --batch-images 16 --crop-size 500 --scale-rate-range 0.7,1.3 --weights models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params --lr-type fixed --base-lr 0.0016 --to-epoch 30 --kvstore local --prefetch-threads 16 --prefetcher process --cache-images 0 --backward-do-mirror --origin-size 1914
~~~~
~~~~
- Train in Cityscapes, please check the [official ResNet-38 repository](https://github.com/itijyou/ademxapp).

### Note
- This code is based on [ResNet-38](https://github.com/itijyou/ademxapp).
- Due to the randomness, the self-training results may slightly vary in each run. Usually the best results will be obtained in 2nd/3rd round. For training in source domain, the best model usually appears during the first 30 epoches. Optimal model appearing in initial stage is also possible.

Contact: yzou2@andrew.cmu.edu
