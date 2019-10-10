import argparse
import sys
from packaging import version
import time
import util
import os
import os.path as osp
import timeit
from collections import OrderedDict
import scipy.io

import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from operator import itemgetter

import scipy
from scipy import ndimage
import math
from PIL import Image
import numpy as np
import shutil
import random

from deeplab.model import Res_Deeplab
from deeplab.datasets import GTA5TestDataSet
from deeplab.datasets import SrcSTDataSet, GTA5StMineDataSet, SoftSrcSTDataSet, SoftGTA5StMineDataSet

### shared ###
IMG_MEAN = np.array((0.406, 0.456, 0.485), dtype=np.float32) # BGR
IMG_STD = np.array((0.225, 0.224, 0.229), dtype=np.float32) # BGR
# data
### source
## gta
DATA_SRC_DIRECTORY = './dataset/gta5'
DATA_SRC_LIST_PATH = './dataset/list/gta5/train.lst'
DATA_SRC = 'gta'
RESTORE_FROM = './src_model/gta5/src_model.pth'
NUM_CLASSES = 19
INIT_SRC_PORT = 0.03 # GTA: 0.03
### target
DATA_TGT_DIRECTORY = './dataset/cityscapes'
DATA_TGT_TRAIN_LIST_PATH = './dataset/list/cityscapes/train_ClsConfSet.lst'
DATA_TGT_TEST_LIST_PATH = './dataset/list/cityscapes/val.lst'
IGNORE_LABEL = 255
# train scales for src and tgt
TRAIN_SCALE_SRC = '0.5,1.5'
TRAIN_SCALE_TGT = '0.5,1.5'
# model
MODEL = 'DeeplabRes'
# gpu
GPU = 0
PIN_MEMORY = False
# log files
LOG_FILE = 'self_training_log'

### train ###
BATCH_SIZE = 2
INPUT_SIZE = '512,1024'# 512,1024 for GTA;
RANDSEED = 3
# params for optimizor
LEARNING_RATE =5e-5
POWER = 0.0
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_ROUNDS = 4
EPR = 2
SRC_SAMPLING_POLICY = 'r'
KC_POLICY = 'cb'
KC_VALUE = 'conf'
INIT_TGT_PORT = 0.2
MAX_TGT_PORT = 0.5
TGT_PORT_STEP = 0.05
# varies but dataset
MAX_SRC_PORT = 0.06 #0.06;
SRC_PORT_STEP = 0.0025 #0.0025:
MRKLD = 0.0
LRENT = 0.0
MRSRC = 0.0
MINE_PORT = 1e-3
RARE_CLS_NUM = 3
MINE_CHANCE = 0.8
### val ###
SAVE_PATH = 'debug'
TEST_IMAGE_SIZE = '1024,2048'
EVAL_SCALE = 0.9
TEST_SCALE = '0.9,1.0,1.2'
DS_RATE = 4

def seed_torch(seed=0):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
   #torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.enabled = False
   #torch.backends.cudnn.deterministic = True

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    ### shared by train & val
    # data
    parser.add_argument("--data-src", type=str, default=DATA_SRC,
                        help="Name of source dataset.")
    parser.add_argument("--data-src-dir", type=str, default=DATA_SRC_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-src-list", type=str, default=DATA_SRC_LIST_PATH,
                        help="Path to the file listing the images&labels in the source dataset.")
    parser.add_argument("--data-tgt-dir", type=str, default=DATA_TGT_DIRECTORY,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-tgt-train-list", type=str, default=DATA_TGT_TRAIN_LIST_PATH,
                        help="Path to the file listing the images*GT labels in the target train dataset.")
    parser.add_argument("--data-tgt-test-list", type=str, default=DATA_TGT_TEST_LIST_PATH,
                        help="Path to the file listing the images*GT labels in the target test dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    # model
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    # gpu
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--pin-memory", type=bool, default=PIN_MEMORY,
                        help="Whether to pin memory in train & eval.")
    # log files
    parser.add_argument("--log-file", type=str, default=LOG_FILE,
                        help="The name of log file.")
    parser.add_argument('--debug',help='True means logging debug info.',
                        default=False, action='store_true')
    ### train ###
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--eval-training", action="store_true",
                        help="Use the saved means and variances, or running means and variances during the evaluation.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--train-scale-src", type=str, default=TRAIN_SCALE_SRC,
                        help="The scale for multi-scale training in source domain.")
    parser.add_argument("--train-scale-tgt", type=str, default=TRAIN_SCALE_TGT,
                    help="The scale for multi-scale training in target domain.")
    # params for optimizor
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    ### val
    parser.add_argument('--test-flipping', dest='test_flipping',
                        help='If average predictions of original and flipped images.',
                        default=False, action='store_true')
    parser.add_argument("--test-image-size", type=str, default=TEST_IMAGE_SIZE,
                        help="The test image size.")
    parser.add_argument("--eval-scale", type=float, default=EVAL_SCALE,
                        help="The test image scale.")
    parser.add_argument("--test-scale", type=str, default=TEST_SCALE,
                        help="The test image scale.")
    ### self-training params
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result for self-training.")
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS,
                        help="Number of rounds for self-training.")
    parser.add_argument("--epr", type=int, default=EPR,
                        help="Number of epochs per round for self-training.")
    parser.add_argument('--kc-policy', default=KC_POLICY, type=str, dest='kc_policy',
                        help='The policy to determine kc. "cb" for weighted class-balanced threshold')
    parser.add_argument('--kc-value', default=KC_VALUE, type=str,
                        help='The way to determine kc values, either "conf", or "prob".')
    parser.add_argument('--ds-rate', default=DS_RATE, type=int,
                        help='The downsampling rate in kc calculation.')
    parser.add_argument('--init-tgt-port', default=INIT_TGT_PORT, type=float, dest='init_tgt_port',
                        help='The initial portion of target to determine kc')
    parser.add_argument('--max-tgt-port', default=MAX_TGT_PORT, type=float, dest='max_tgt_port',
                        help='The max portion of target to determine kc')
    parser.add_argument('--tgt-port-step', default=TGT_PORT_STEP, type=float, dest='tgt_port_step',
                        help='The portion step in target domain in every round of self-paced self-trained neural network')
    parser.add_argument('--init-src-port', default=INIT_SRC_PORT, type=float, dest='init_src_port',
                        help='The initial portion of source portion for self-trained neural network')
    parser.add_argument('--max-src-port', default=MAX_SRC_PORT, type=float, dest='max_src_port',
                        help='The max portion of source portion for self-trained neural network')
    parser.add_argument('--src-port-step', default=SRC_PORT_STEP, type=float, dest='src_port_step',
                        help='The portion step in source domain in every round of self-paced self-trained neural network')
    parser.add_argument('--randseed', default=RANDSEED, type=int,
                        help='The random seed to sample the source dataset.')
    parser.add_argument("--src-sampling-policy", type=str, default=SRC_SAMPLING_POLICY,
                        help="The sampling policy on source dataset: 'c' for 'cumulative' and 'r' for replace ")
    parser.add_argument('--mine-port', default=MINE_PORT, type=float,
                        help='If a class has a predication portion lower than the mine_port, then mine the patches including the class in self-training.')
    parser.add_argument('--rare-cls-num', default=RARE_CLS_NUM, type=int,
                        help='The number of classes to be mined.')
    parser.add_argument('--mine-chance', default=MINE_CHANCE, type=float,
                        help='The chance of patch mining.')
    parser.add_argument('--rm-prob',
                        help='If remove the probability maps generated in every round.',
                        default=False, action='store_true')
    parser.add_argument('--mr-weight-kld', default=MRKLD, type=float, dest='mr_weight_kld',
                        help='weight of kld model regularization')
    parser.add_argument('--lr-weight-ent', default=LRENT, type=float, dest='lr_weight_ent',
                        help='weight of negative entropy label regularization')
    parser.add_argument('--mr-weight-src', default=MRSRC, type=float, dest='mr_weight_src',
                        help='weight of regularization in source domain')
    return parser.parse_args()

args = get_arguments()

# palette
if args.data_src == 'gta':
    # gta:
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def main():
    randseed = args.randseed
    seed_torch(randseed)
    device = torch.device("cuda:" + str(args.gpu))
    save_path = args.save
    save_pseudo_label_path = osp.join(save_path, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    save_stats_path = osp.join(save_path, 'stats') # in 'save_path'
    save_lst_path = osp.join(save_path, 'list')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path)
    if not os.path.exists(save_stats_path):
        os.makedirs(save_stats_path)
    if not os.path.exists(save_lst_path):
        os.makedirs(save_lst_path)

    logger = util.set_logger(args.save, args.log_file, args.debug)
    logger.info('start with arguments %s', args)

    if args.model == 'DeeplabRes':
        model = Res_Deeplab(num_classes=args.num_classes)


    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = str(i).split('.')
            # print i_parts
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    else:
	loc = "cuda:" + str(args.gpu)
        saved_state_dict = torch.load(args.restore_from,map_location=loc)
        new_params = saved_state_dict.copy()

    model.load_state_dict(new_params)

    image_src_list, _, label_src_list, src_num = parse_split_list(args.data_src_list)
    image_tgt_list, image_name_tgt_list, _, tgt_num = parse_split_list(args.data_tgt_train_list)
    _, _, _, test_num = parse_split_list(args.data_tgt_test_list)

    ## label mapping
    sys.path.insert(0, 'dataset/helpers')
    if args.data_src == 'synthia':
        from labels_cityscapes_synthia import id2label, trainId2label
    elif args.data_src == 'gta':
        from labels import id2label, trainId2label
    label_2_id = 255 * np.ones((256,))
    for l in id2label:
        if l in (-1, 255):
            continue
        label_2_id[l] = id2label[l].trainId
    id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
    valid_labels = sorted(set(id_2_label.ravel()))

    # portions
    tgt_portion = args.init_tgt_port
    src_portion = args.init_src_port

    # training crop size
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    lscale_src, hscale_src = map(float, args.train_scale_src.split(','))
    train_scale_src = (lscale_src, hscale_src)
    lscale_tgt, hscale_tgt = map(float, args.train_scale_tgt.split(','))
    train_scale_tgt = (lscale_tgt, hscale_tgt)

    for round_idx in range(args.num_rounds):
        save_round_eval_path = osp.join(args.save,str(round_idx))
        save_pseudo_label_color_path = osp.join(save_round_eval_path, 'pseudo_label_color')  # in every 'save_round_eval_path'
        if not os.path.exists(save_round_eval_path):
            os.makedirs(save_round_eval_path)
        if not os.path.exists(save_pseudo_label_color_path):
            os.makedirs(save_pseudo_label_color_path)
        ########## pseudo-label generation
        if round_idx != args.num_rounds - 1:
            # evaluation & save confidence vectors
            conf_dict, pred_cls_num, save_prob_path, save_pred_path = val(model, device, save_round_eval_path, round_idx, tgt_num,
                                                            label_2_id, valid_labels, args, logger)
            # class-balanced thresholds
            cls_thresh = kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, args, logger)
            tgt_portion = min(tgt_portion + args.tgt_port_step, args.max_tgt_port)
            # pseudo-label maps generation
            label_selection(cls_thresh, tgt_num, image_name_tgt_list, id_2_label, round_idx, save_prob_path, save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args, logger)
            # save training list
            if args.src_sampling_policy == 'c':
                randseed = args.randseed
            elif args.src_sampling_policy == 'r':
                randseed += 1
            src_train_lst, tgt_train_lst, src_num_sel = savelst_SrcTgt(src_portion, image_tgt_list, image_name_tgt_list, image_src_list, label_src_list, save_lst_path, save_pseudo_label_path, src_num, tgt_num, randseed, args)
            src_portion = min(src_portion + args.src_port_step, args.max_src_port)
            ########### model retraining
            # dataset
            epoch_per_round = args.epr
            # reg weights
            if args.mr_weight_kld == 0:
                reg_weight_tgt = 0.0
            else:  # currently only one kind of model regularizer is supported
                reg_weight_tgt = args.mr_weight_kld
            reg_weight_src = args.mr_weight_src
            ### patch mining params
            # no patch mining in src
            # patch mining in target
            rare_id = np.load(save_stats_path + '/rare_id_round' + str(round_idx) + '.npy')
            mine_id = np.load(save_stats_path + '/mine_id_round' + str(round_idx) + '.npy')
            mine_chance = args.mine_chance
            # dataloader
            if args.lr_weight_ent == 0.0:
                srctrainset = SrcSTDataSet(args.data_src_dir, src_train_lst, max_iters=src_num_sel,reg_weight=reg_weight_src,data_src=args.data_src,
                                          crop_size=input_size,scale=args.random_scale, mirror=args.random_mirror, train_scale=train_scale_src, mean=IMG_MEAN, std=IMG_STD)
                tgttrainset = GTA5StMineDataSet(args.data_tgt_dir, tgt_train_lst, pseudo_root=save_pseudo_label_path, max_iters=tgt_num,reg_weight=reg_weight_tgt,rare_id = rare_id,
                                          mine_id=mine_id, mine_chance = mine_chance, crop_size=input_size,scale=args.random_scale,data_src=args.data_src,
                                          mirror=args.random_mirror, train_scale=train_scale_tgt, mean=IMG_MEAN, std=IMG_STD)
            elif args.lr_weight_ent > 0.0:
                srctrainset = SoftSrcSTDataSet(args.data_src_dir, src_train_lst, max_iters=src_num_sel,reg_weight=reg_weight_src,data_src=args.data_src,num_classes=args.num_classes,
                                          crop_size=input_size,scale=args.random_scale, mirror=args.random_mirror, train_scale=train_scale_src, mean=IMG_MEAN, std=IMG_STD)
                tgttrainset = SoftGTA5StMineDataSet(args.data_tgt_dir, tgt_train_lst, pseudo_root=save_pseudo_label_path, max_iters=tgt_num,reg_weight=reg_weight_tgt,rare_id = rare_id,
                                          mine_id=mine_id, mine_chance = mine_chance, crop_size=input_size,scale=args.random_scale,data_src=args.data_src,num_classes=args.num_classes,
                                          mirror=args.random_mirror, train_scale=train_scale_tgt, mean=IMG_MEAN, std=IMG_STD)
            mixtrainset = torch.utils.data.ConcatDataset([srctrainset, tgttrainset])
            mix_trainloader = torch.utils.data.DataLoader(mixtrainset, batch_size=args.batch_size, shuffle=True,
                                                          num_workers=0, pin_memory=args.pin_memory)
            # optimizer
            tot_iter = np.ceil(float(src_num_sel + tgt_num) / args.batch_size)
            optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate},
                                   {'params': get_10x_lr_params(model), 'lr': 10 * args.learning_rate}],
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            logger.info('###### Start model retraining dataset in round {}! ######'.format(round_idx))

            # model
            if args.is_training:
                model.train()
            else:
                model.eval()
            start = timeit.default_timer()

            # cudnn
            cudnn.enabled = True  # enable cudnn
            cudnn.benchmark = True  # enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

            # start training
            for epoch in range(epoch_per_round):
                train(mix_trainloader, model, device, interp, optimizer, tot_iter, round_idx, epoch, args, logger)
            end = timeit.default_timer()
            logger.info('###### Finish model retraining dataset in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx, end - start))

            # test self-trained model in target domain test set
            tgt_set = 'test'
            test(model, device, save_round_eval_path, round_idx, tgt_set, test_num, args.data_tgt_test_list, label_2_id,
                 valid_labels, args, logger)
        elif round_idx == args.num_rounds - 1:
            shutil.rmtree(save_pseudo_label_path)
            tgt_set = 'train'
            test(model, device, save_round_eval_path, round_idx, tgt_set, tgt_num, args.data_tgt_train_list, label_2_id,
                 valid_labels, args, logger)
            tgt_set = 'test'
            test(model, device, save_round_eval_path, round_idx, tgt_set, test_num, args.data_tgt_test_list, label_2_id,
                 valid_labels, args, logger)

def val(model, device, save_round_eval_path, round_idx, tgt_num, label_2_id, valid_labels, args, logger):
    """Create the model and start the evaluation process."""
    ## scorer
    scorer = ScoreUpdater(valid_labels, args.num_classes, tgt_num, logger)
    scorer.reset()
    h, w = map(int, args.test_image_size.split(','))
    test_image_size = (h, w)
    test_size = ( int(h*args.eval_scale), int(w*args.eval_scale) )

    ## test data loader
    testloader = data.DataLoader(GTA5TestDataSet(args.data_tgt_dir, args.data_tgt_train_list, test_size=test_size, test_scale=args.eval_scale, mean=IMG_MEAN, std=IMG_STD, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=args.pin_memory)

    ## model for evaluation
    if args.eval_training:
        model.train()
    else:
        model.eval()
    #
    model.to(device)

    ## upsampling layer
    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=test_image_size, mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=test_image_size, mode='bilinear')

    ## output of deeplab is logits, not probability
    softmax2d = nn.Softmax2d()

    ## output folder
    save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
    save_prob_path = osp.join(save_round_eval_path, 'prob')
    save_pred_path = osp.join(save_round_eval_path, 'pred')
    if not os.path.exists(save_pred_vis_path):
        os.makedirs(save_pred_vis_path)
    if not os.path.exists(save_prob_path):
        os.makedirs(save_prob_path)
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)

    # saving output data
    conf_dict = {k: [] for k in range(args.num_classes)}
    pred_cls_num = np.zeros(args.num_classes)
    ## evaluation process
    logger.info('###### Start evaluating target domain train set in round {}! ######'.format(round_idx))
    start_eval = time.time()
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            image, label, _, name = batch
            if args.model == 'DeeplabRes':
                output2 = model(image.to(device))
                output = softmax2d(interp(output2)).cpu().data[0].numpy()
            if args.test_flipping:
                output2 = model(torch.from_numpy(image.numpy()[:,:,:,::-1].copy()).to(device))
                output = 0.5 * ( output + softmax2d(interp(output2)).cpu().data[0].numpy()[:,:,::-1] )
            output = output.transpose(1,2,0)
            amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            conf = np.amax(output,axis=2)
            # score
            pred_label = amax_output.copy()
            label = label_2_id[np.asarray(label.numpy(), dtype=np.uint8)]
            scorer.update(pred_label.flatten(), label.flatten(), index)

            # save visualized seg maps & predication prob map
            amax_output_col = colorize_mask(amax_output)
            name = name[0].split('/')[-1]
            image_name = name.split('.')[0]
            # prob
            np.save('%s/%s.npy' % (save_prob_path, image_name), output)
            # trainIDs/vis seg maps
            amax_output = Image.fromarray(amax_output)
            amax_output.save('%s/%s.png' % (save_pred_path, image_name))
            amax_output_col.save('%s/%s_color.png' % (save_pred_vis_path, image_name))

            # save class-wise confidence maps
            if args.kc_value == 'conf':
                for idx_cls in range(args.num_classes):
                    idx_temp = pred_label == idx_cls
                    pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
                    if idx_temp.any():
                        conf_cls_temp = conf[idx_temp].astype(np.float32)
                        len_cls_temp = conf_cls_temp.size
                        # downsampling by ds_rate
                        conf_cls = conf_cls_temp[0:len_cls_temp:args.ds_rate]
                        conf_dict[idx_cls].extend(conf_cls)
            elif args.kc_value == 'prob':
                for idx_cls in range(args.num_classes):
                    idx_temp = pred_label == idx_cls
                    pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
                    # prob slice
                    prob_cls_temp = output[:,:,idx_cls].astype(np.float32).ravel()
                    len_cls_temp = prob_cls_temp.size
                    # downsampling by ds_rate
                    prob_cls = prob_cls_temp[0:len_cls_temp:args.ds_rate]
                    conf_dict[idx_cls].extend(prob_cls) # it should be prob_dict; but for unification, use conf_dict
    logger.info('###### Finish evaluating target domain train set in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx, time.time()-start_eval))

    return conf_dict, pred_cls_num, save_prob_path, save_pred_path  # return the dictionary containing all the class-wise confidence vectors

def train(mix_trainloader, model, device, interp, optimizer, tot_iter, round_idx, epoch_idx, args, logger):
    """Create the model and start the training."""
    for i_iter, batch in enumerate(mix_trainloader):
        images, labels, _, _, reg_weights = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, tot_iter)
        pred = interp(model(images))
        if args.lr_weight_ent == 0.0:
            loss = reg_loss_calc(pred, labels, reg_weights.to(device), args)
        if args.lr_weight_ent > 0.0:
            loss = reg_loss_calc_expand(pred, labels, reg_weights.to(device), args)
        loss.backward()
        optimizer.step()

        logger.info('iter = {} of {} completed, loss = {:.4f}'.format(i_iter+1, tot_iter, loss.data.cpu().numpy()))

    print('taking snapshot ...')
    torch.save(model.state_dict(), osp.join(args.save, args.data_src + '2city_round' + str(round_idx) + '_epoch' + str(epoch_idx+1)  + '.pth'))

def test(model, device, save_round_eval_path, round_idx, tgt_set, test_num, test_list, label_2_id, valid_labels, args, logger):
    """Create the model and start the evaluation process."""
    ## scorer
    scorer = ScoreUpdater(valid_labels, args.num_classes, test_num, logger)
    scorer.reset()
    h, w = map(int, args.test_image_size.split(','))
    test_image_size = (h, w)
    test_size = ( h, w )
    test_scales = [float(_) for _ in str(args.test_scale).split(',')]
    num_scales = len(test_scales)

    ## test data loader
    testloader = data.DataLoader(GTA5TestDataSet(args.data_tgt_dir, test_list, test_size=test_size, test_scale=1.0, mean=IMG_MEAN, std=IMG_STD, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=args.pin_memory)

    ## model for evaluation
    if args.eval_training:
        model.train()
    else:
        model.eval()
    #
    model.to(device)

    ## upsampling layer
    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=test_image_size, mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=test_image_size, mode='bilinear')

    ## output of deeplab is logits, not probability
    softmax2d = nn.Softmax2d()

    ## output folder
    if tgt_set == 'train':
        save_test_vis_path = osp.join(save_round_eval_path, 'trainSet_vis')
    elif tgt_set == 'test':
        save_test_vis_path = osp.join(save_round_eval_path, 'testSet_vis')
    if not os.path.exists(save_test_vis_path):
        os.makedirs(save_test_vis_path)

    ## evaluation process
    logger.info('###### Start evaluating in target domain {} set in round {}! ######'.format(tgt_set, round_idx))
    start_eval = time.time()
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            image, label, _, name = batch
            img = image.clone()
            for scale_idx in range(num_scales):
                if version.parse(torch.__version__) > version.parse('0.4.0'):
                    image = F.interpolate(img, scale_factor=test_scales[scale_idx], mode='bilinear', align_corners=True)
                else:
                    test_size = (int(h * test_scales[scale_idx]), int(w * test_scales[scale_idx]))
                    interp_tmp = nn.Upsample(size=test_size, mode='bilinear', align_corners=True)
                    image = interp_tmp(img)
                if args.model == 'DeeplabRes':
                    output2 = model(image.to(device))
                    coutput = interp(output2).cpu().data[0].numpy()
                if args.test_flipping:
                    output2 = model(torch.from_numpy(image.numpy()[:,:,:,::-1].copy()).to(device))
                    coutput = 0.5 * ( coutput + interp(output2).cpu().data[0].numpy()[:,:,::-1] )
                if scale_idx == 0:
                    output = coutput.copy()
                else:
                    output = output+coutput
            output = output/num_scales
            output = output.transpose(1,2,0)
            amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            # score
            pred_label = amax_output.copy()
            label = label_2_id[np.asarray(label.numpy(), dtype=np.uint8)]
            scorer.update(pred_label.flatten(), label.flatten(), index)
            # save visualized seg maps & predication prob map
            amax_output_col = colorize_mask(amax_output)
            name = name[0].split('/')[-1]
            image_name = name.split('.')[0]
            # vis seg maps
            amax_output_col.save('%s/%s_color.png' % (save_test_vis_path, image_name))

    logger.info('###### Finish evaluating in target domain {} set in round {}! Time cost: {:.2f} seconds. ######'.format(tgt_set, round_idx, time.time()-start_eval))
    return

def kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, args, logger):
    logger.info('###### Start kc generation in round {} ! ######'.format(round_idx))
    start_kc = time.time()
    # threshold for each class
    conf_tot = 0.0
    cls_thresh = np.ones(args.num_classes,dtype = np.float32)
    cls_sel_size = np.zeros(args.num_classes, dtype=np.float32)
    cls_size = np.zeros(args.num_classes, dtype=np.float32)
    if args.kc_policy == 'cb' and args.kc_value == 'conf':
        for idx_cls in np.arange(0, args.num_classes):
            cls_size[idx_cls] = pred_cls_num[idx_cls]
            if conf_dict[idx_cls] != None:
                conf_dict[idx_cls].sort(reverse=True) # sort in descending order
                len_cls = len(conf_dict[idx_cls])
                cls_sel_size[idx_cls] = int(math.floor(len_cls * tgt_portion))
                len_cls_thresh = int(cls_sel_size[idx_cls])
                if len_cls_thresh != 0:
                    cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
                conf_dict[idx_cls] = None

    # threshold for mine_id with priority
    num_mine_id = len(np.nonzero(cls_size / np.sum(cls_size) < args.mine_port)[0])
    # chose the smallest mine_id
    id_all = np.argsort(cls_size / np.sum(cls_size))
    rare_id = id_all[:args.rare_cls_num]
    mine_id = id_all[:num_mine_id] # sort mine_id in ascending order w.r.t predication portions
    # save mine ids
    np.save(save_stats_path + '/rare_id_round' + str(round_idx) + '.npy', rare_id)
    np.save(save_stats_path + '/mine_id_round' + str(round_idx) + '.npy', mine_id)
    logger.info('Mining ids : {}! {} rarest ids: {}!'.format(mine_id,args.rare_cls_num,rare_id))
    # save thresholds
    np.save(save_stats_path + '/cls_thresh_round' + str(round_idx) + '.npy', cls_thresh)
    np.save(save_stats_path + '/cls_sel_size_round' + str(round_idx) + '.npy', cls_sel_size)
    logger.info('###### Finish kc generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx,time.time() - start_kc))
    return cls_thresh

def label_selection(cls_thresh, tgt_num, image_name_tgt_list, id_2_label, round_idx, save_prob_path, save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args, logger):
    logger.info('###### Start pseudo-label generation in round {} ! ######'.format(round_idx))
    start_pl = time.time()
    for idx in range(tgt_num):
        sample_name = image_name_tgt_list[idx].split('.')[0]
        probmap_path = osp.join(save_prob_path, '{}.npy'.format(sample_name))
        pred_path = osp.join(save_pred_path, '{}.png'.format(sample_name))
        pred_prob = np.load(probmap_path)
        pred_label_trainIDs = np.asarray(Image.open(pred_path))
        pred_label_labelIDs = id_2_label[pred_label_trainIDs]
        pred_label_trainIDs = pred_label_trainIDs.copy()
        if args.kc_policy == 'cb' and args.lr_weight_ent == 0.0:
            save_wpred_vis_path = osp.join(save_round_eval_path, 'weighted_pred_vis')
            if not os.path.exists(save_wpred_vis_path):
                os.makedirs(save_wpred_vis_path)
            weighted_prob = pred_prob/cls_thresh
            weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob, axis=2), dtype=np.uint8)
            # save weighted predication
            wpred_label_col = weighted_pred_trainIDs.copy()
            wpred_label_col = colorize_mask(wpred_label_col)
            wpred_label_col.save('%s/%s_color.png' % (save_wpred_vis_path, sample_name))
            weighted_conf = np.amax(weighted_prob, axis=2)
            pred_label_trainIDs = weighted_pred_trainIDs.copy()
            pred_label_labelIDs = id_2_label[pred_label_trainIDs]
            pred_label_labelIDs[weighted_conf < 1] = 0  # '0' in cityscapes indicates 'unlabaled' for labelIDs
            pred_label_trainIDs[weighted_conf < 1] = 255 # '255' in cityscapes indicates 'unlabaled' for trainIDs
        elif args.kc_policy == 'cb' and args.lr_weight_ent > 0.0: # check if cb can be combined with kc_value == conf or prob; also check if \alpha can be larger than 1
            save_wpred_vis_path = osp.join(save_round_eval_path, 'weighted_pred_vis')
            if not os.path.exists(save_wpred_vis_path):
                os.makedirs(save_wpred_vis_path)
            # soft pseudo-label
            soft_pseudo_label = np.power(pred_prob/cls_thresh,1.0/args.lr_weight_ent) # weighted softmax with temperature
            soft_pseudo_label_sum = soft_pseudo_label.sum(2)
            soft_pseudo_label = soft_pseudo_label.transpose(2,0,1)/soft_pseudo_label_sum
            soft_pseudo_label = soft_pseudo_label.transpose(1,2,0).astype(np.float32)
            np.save('%s/%s.npy' % (save_pseudo_label_path, sample_name), soft_pseudo_label)
            # hard pseudo-label
            weighted_pred_trainIDs = np.asarray(np.argmax(soft_pseudo_label, axis=2), dtype=np.uint8)
            reg_score = np.sum( -soft_pseudo_label*np.log(pred_prob+1e-32) + args.lr_weight_ent*soft_pseudo_label*np.log(soft_pseudo_label+1e-32), axis=2)
            sel_score = np.sum( -soft_pseudo_label*np.log(cls_thresh+1e-32), axis=2)
            # save weighted predication
            wpred_label_col = weighted_pred_trainIDs.copy()
            wpred_label_col = colorize_mask(wpred_label_col)
            wpred_label_col.save('%s/%s_color.png' % (save_wpred_vis_path, sample_name))
            pred_label_trainIDs = weighted_pred_trainIDs.copy()
            pred_label_labelIDs = id_2_label[pred_label_trainIDs]
            pred_label_labelIDs[reg_score >= sel_score] = 0  # '0' in cityscapes indicates 'unlabaled' for labelIDs
            pred_label_trainIDs[reg_score >= sel_score] = 255 # '255' in cityscapes indicates 'unlabaled' for trainIDs

        # pseudo-labels with labelID
        pseudo_label_labelIDs = pred_label_labelIDs.copy()
        pseudo_label_trainIDs = pred_label_trainIDs.copy()
        # save colored pseudo-label map
        pseudo_label_col = colorize_mask(pseudo_label_trainIDs)
        pseudo_label_col.save('%s/%s_color.png' % (save_pseudo_label_color_path, sample_name))
        # save pseudo-label map with label IDs
        pseudo_label_save = Image.fromarray(pseudo_label_labelIDs.astype(np.uint8))
        pseudo_label_save.save('%s/%s.png' % (save_pseudo_label_path, sample_name))

    # remove probability maps
    if args.rm_prob:
        shutil.rmtree(save_prob_path)

    logger.info('###### Finish pseudo-label generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx,time.time() - start_pl))

def parse_split_list(list_name):
    image_list = []
    image_name_list = []
    label_list = []
    file_num = 0
    with open(list_name) as f:
        for item in f.readlines():
            fields = item.strip().split('\t')
            image_name = fields[0].split('/')[-1]
            image_list.append(fields[0])
            image_name_list.append(image_name)
            label_list.append(fields[1])
            file_num += 1
    return image_list, image_name_list, label_list, file_num

def savelst_SrcTgt(src_portion, image_tgt_list, image_name_tgt_list, image_src_list, label_src_list, save_lst_path, save_pseudo_label_path, src_num, tgt_num, randseed, args):
    src_num_sel = int(np.floor(src_num*src_portion))
    np.random.seed(randseed)
    sel_idx = list( np.random.choice(src_num, src_num_sel, replace=False) )
    sel_src_img_list = list( itemgetter(*sel_idx)(image_src_list) )
    sel_src_label_list = list(itemgetter(*sel_idx)(label_src_list))
    src_train_lst = osp.join(save_lst_path,'src_train.lst')
    tgt_train_lst = osp.join(save_lst_path, 'tgt_train.lst')

    # generate src train list
    with open(src_train_lst, 'w') as f:
        for idx in range(src_num_sel):
            f.write("%s\t%s\n" % (sel_src_img_list[idx], sel_src_label_list[idx]))
    # generate tgt train list
    if args.lr_weight_ent > 0:
        with open(tgt_train_lst, 'w') as f:
            for idx in range(tgt_num):
                softlabel_name = image_name_tgt_list[idx].split('.')[0] + '.npy'
                soft_label_tgt_path = osp.join(save_pseudo_label_path, softlabel_name)
                image_tgt_path = osp.join(save_pseudo_label_path,image_name_tgt_list[idx])
                f.write("%s\t%s\t%s\n" % (image_tgt_list[idx], image_tgt_path, soft_label_tgt_path))
    elif args.lr_weight_ent == 0:
        with open(tgt_train_lst, 'w') as f:
            for idx in range(tgt_num):
                image_tgt_path = osp.join(save_pseudo_label_path,image_name_tgt_list[idx])
                f.write("%s\t%s\n" % (image_tgt_list[idx], image_tgt_path))

    return src_train_lst, tgt_train_lst, src_num_sel

class ScoreUpdater(object):
    # only IoU are computed. accu, cls_accu, etc are ignored.
    def __init__(self, valid_labels, c_num, x_num, logger=None, label=None, info=None):
        self._valid_labels = valid_labels

        self._confs = np.zeros((c_num, c_num))
        self._per_cls_iou = np.zeros(c_num)
        self._logger = logger
        self._label = label
        self._info = info
        self._num_class = c_num
        self._num_sample = x_num

    @property
    def info(self):
        return self._info

    def reset(self):
        self._start = time.time()
        self._computed = np.zeros(self._num_sample) # one-dimension
        self._confs[:] = 0

    def fast_hist(self,label, pred_label, n):
        k = (label >= 0) & (label < n)
        return np.bincount(n * label[k].astype(int) + pred_label[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(self,hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def do_updates(self, conf, i, computed=True):
        if computed:
            self._computed[i] = 1
        self._per_cls_iou = self.per_class_iu(conf)

    def update(self, pred_label, label, i, computed=True):
        conf = self.fast_hist(label, pred_label, self._num_class)
        self._confs += conf
        self.do_updates(self._confs, i, computed)
        self.scores(i)

    def scores(self, i=None, logger=None):
        x_num = self._num_sample
        ious = np.nan_to_num( self._per_cls_iou )

        logger = self._logger if logger is None else logger
        if logger is not None:
            if i is not None:
                speed = 1. * self._computed.sum() / (time.time() - self._start)
                logger.info('Done {}/{} with speed: {:.2f}/s'.format(i + 1, x_num, speed))
            name = '' if self._label is None else '{}, '.format(self._label)
            logger.info('{}mean iou: {:.2f}%'. \
                        format(name, np.mean(ious) * 100))
            with util.np_print_options(formatter={'float': '{:5.2f}'.format}):
                logger.info('\n{}'.format(ious * 100))

        return ious

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL).cuda()

    return criterion(pred, label)

def reg_loss_calc(pred, label, reg_weights, args):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    mr_weight_kld = args.mr_weight_kld
    num_class = float(args.num_classes)
    valid_num = torch.sum(label != IGNORE_LABEL).float()

    label_reg = label[reg_weights != 0,:,:]
    valid_reg_num = torch.sum(label_reg != IGNORE_LABEL).float()

    softmax = F.softmax(pred, dim=1)   # compute the softmax values
    logsoftmax = F.log_softmax(pred,dim=1)   # compute the log of softmax values

    label_expand = torch.unsqueeze(label, 1).repeat(1,int(num_class),1,1)
    labels = label_expand.clone()
    labels[labels != IGNORE_LABEL] = 1.0
    labels[labels == IGNORE_LABEL] = 0.0
    labels_valid = labels.clone()
    # labels = torch.unsqueeze(labels, 1).repeat(1,num_class,1,1)
    labels = torch.cumsum(labels, dim=1)
    labels[labels != label_expand + 1] = 0.0
    del label_expand
    labels[labels != 0 ] = 1.0
    ### check the vectorized labels
    # check_labels = torch.argmax(labels, dim=1)
    # label[label == 255] = 0
    # print(torch.sum(check_labels.float() - label))
    reg_weights = reg_weights.float().view(len(reg_weights),1,1,1)
    ce = torch.sum( -logsoftmax*labels ) # cross-entropy loss with vector-form softmax
    softmax_val = softmax*labels_valid
    logsoftmax_val = logsoftmax*labels_valid
    kld = torch.sum( -logsoftmax_val/num_class*reg_weights )

    if valid_reg_num > 0:
        reg_ce = ce/valid_num + (mr_weight_kld*kld)/valid_reg_num
    else:
        reg_ce = ce/valid_num

    return reg_ce

def reg_loss_calc_expand(pred, label, reg_weights, args):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    mr_weight_kld = args.mr_weight_kld
    num_class = float(args.num_classes)
    # soft labels regard ignored labels as zero soft labels in data loader
    # C = label.cpu().numpy()
    label_sum = torch.sum(label,1)
    # D = label_sum.cpu().numpy()
    valid_num = torch.sum(label_sum != 0.0).float()

    label_reg = label_sum[reg_weights != 0,:,:]
    valid_reg_num = torch.sum(label_reg != 0.0).float()

    softmax = F.softmax(pred, dim=1)   # compute the softmax values
    logsoftmax = F.log_softmax(pred,dim=1)   # compute the log of softmax values

    label_expand = torch.unsqueeze(label_sum, 1).repeat(1,num_class,1,1)
    label_valid = label_expand.clone()
    label_valid[label_valid != 0] = 1.0
    label_valid = label_valid.clone()
    # # check the vectorized labels
    # check_labels = torch.argmax(labels, dim=1)
    # label[label == 255] = 0
    # print(torch.sum(check_labels.float() - label))
    #
    reg_weights = reg_weights.float().view(len(reg_weights),1,1,1)
    ce = torch.sum( -logsoftmax*label ) # cross-entropy loss with vector-form softmax
    softmax_val = softmax*label_valid
    logsoftmax_val = logsoftmax*label_valid
    kld = torch.sum( -logsoftmax_val/num_class*reg_weights )

    if valid_reg_num > 0:
        reg_ce = ce/valid_num + (mr_weight_kld*kld)/valid_reg_num
    else:
        reg_ce = ce/valid_num

    return reg_ce

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def adjust_learning_rate(optimizer, i_iter, tot_iter):
    lr = lr_poly(args.learning_rate, i_iter, tot_iter, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10

if __name__ == '__main__':
    main()
