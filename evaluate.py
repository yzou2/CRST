import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version
import time
import util

import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from deeplab.model import Res_Deeplab
from deeplab.datasets import GTA5TestDataSet
from collections import OrderedDict
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IMG_MEAN = np.array((0.406, 0.456, 0.485), dtype=np.float32) # BGR
IMG_STD = np.array((0.225, 0.224, 0.229), dtype=np.float32) # BGR

DATA_DIRECTORY = './dataset/cityscapes'
DATA_LIST_PATH = './dataset/list/cityscapes/val.lst'
SAVE_PATH = './cityscapes/eval'

TEST_IMAGE_SIZE = '1024,2048'
TEST_SCALES = 1.0
IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = './src_model/gta5/src_model.pth'
DATA_SRC = 'cityscapes'
SET = 'val'
LOG_FILE = 'log'

MODEL = 'DeeplabRes'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG).")
    parser.add_argument("--data-src", type=str, default=DATA_SRC,
                        help="Data name.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument('--test-flipping', dest='test_flipping',
                        help='If average predictions of original and flipped images.',
                        default=False, action='store_true')
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--log-file", type=str, default=LOG_FILE,
                        help="The name of log file.")
    parser.add_argument('--debug',help='True means logging debug info.',
                        default=False, action='store_true')
    parser.add_argument('--test-scales', type=str, default=TEST_SCALES,
                        help='The test scales. Multi-scale supported')
    parser.add_argument('--test-image-size', default=TEST_IMAGE_SIZE,
                        help='The test image size',
                        type=str)
    return parser.parse_args()

args = get_arguments()

# palette
if args.data_src == 'gta' or args.data_src == 'cityscapes':
    # gta:
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
elif args.data_src == 'synthia':
    # synthia:
    palette = [128,64,128,244,35,232,70,70,70,102,102,156,64,64,128,153,153,153,250,170,30,220,220,0,
                107,142,35,70,130,180,220,20,60,255,0,0,0,0,142,0,60,100,0,0,230,119,11,32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def main():
    """Create the model and start the evaluation process."""
    device = torch.device("cuda:" + str(args.gpu))

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    logger = util.set_logger(args.save, args.log_file, args.debug)
    logger.info('start with arguments %s', args)

    x_num = 0

    with open(args.data_list) as f:
        for _ in f.readlines():
            x_num = x_num + 1

    sys.path.insert(0, 'dataset/helpers')
    if args.data_src == 'gta' or args.data_src == 'cityscapes':
        from labels import id2label, trainId2label
    elif args.data_src == 'synthia':
        from labels_cityscapes_synthia import id2label, trainId2label
    #
    label_2_id = 255 * np.ones((256,))
    for l in id2label:
        if l in (-1, 255):
            continue
        label_2_id[l] = id2label[l].trainId
    id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
    valid_labels = sorted(set(id_2_label.ravel()))
    scorer = ScoreUpdater(valid_labels, args.num_classes, x_num, logger)
    scorer.reset()

    if args.model == 'DeeplabRes':
        model = Res_Deeplab(num_classes=args.num_classes)
    # elif args.model == 'DeeplabVGG':
    #     model = DeeplabVGG(num_classes=args.num_classes)
    #     if args.restore_from == RESTORE_FROM:
    #         args.restore_from = RESTORE_FROM_VGG

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
    #model.train()
    model.eval()
    model.to(device)

    testloader = data.DataLoader(GTA5TestDataSet(args.data_dir, args.data_list, test_scale = 1.0, test_size=(1024, 512), mean=IMG_MEAN, std=IMG_STD, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=True)

    test_scales = [float(_) for _ in str(args.test_scales).split(',')]

    h, w = map(int, args.test_image_size.split(','))
    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(h, w), mode='bilinear')

    test_image_size = (h, w)
    mean_rgb = IMG_MEAN[::-1].copy()
    std_rgb = IMG_STD[::-1].copy()
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            image, label, _, name = batch
            img = image.clone()
            num_scales = len(test_scales)
            # output_dict = {k: [] for k in range(num_scales)}
            for scale_idx in range(num_scales):
                if version.parse(torch.__version__) > version.parse('0.4.0'):
                    image = F.interpolate(image, scale_factor=test_scales[scale_idx], mode='bilinear', align_corners=True)
                else:
                    test_size = ( int(h*test_scales[scale_idx]), int(w*test_scales[scale_idx]) )
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
                    output += coutput

            output = output/num_scales
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            pred_label = output.copy()
            label = label_2_id[np.asarray(label.numpy(), dtype=np.uint8)]
            scorer.update(pred_label.flatten(), label.flatten(), index)

            output_col = colorize_mask(output)
            output = Image.fromarray(output)

            name = name[0].split('/')[-1]
            output.save('%s/%s' % (args.save, name))
            output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))

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

if __name__ == '__main__':
    main()
