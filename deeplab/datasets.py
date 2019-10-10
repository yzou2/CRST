import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision.transforms as transforms
import torchvision
import cv2
from torch.utils import data
import sys
from PIL import Image

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

class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "img/%s.jpg" % name)
            label_file = osp.join(self.root, "gt/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name

class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, pseudo_root = None, max_iters=None, crop_size=(500, 500), train_scale = (0.5, 1.5), mean=(128, 128, 128), std = (1,1,1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.pseudo_root = pseudo_root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.crop_h, self.crop_w = crop_size
        self.lscale, self.hscale = train_scale
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            if self.pseudo_root == None:
                label_file = osp.join(self.root, label_name)
            else:
                label_file = label_name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):

        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = self.lscale + random.randint(0, int((self.hscale-self.lscale)*10)) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # OpenCV read image as BGR, not RGB
        label = np.array(Image.open(datafiles["label"]))
        #
        sys.path.insert(0, 'dataset/helpers')
        from labels import id2label, trainId2label
        #
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        # id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        # valid_labels = sorted(set(id_2_label.ravel()))
        label = label_2_id[label]
        #
        size = image.shape
        img_name = datafiles["img_name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = image/255.0 # scale to [0,1]
        image -= self.mean # BGR
        image = image/self.std#np.reshape(self.std,(1,1,3))

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), img_name


class SYNTHIADataSet(data.Dataset):
    def __init__(self, root, list_path, pseudo_root = None, max_iters=None, crop_size=(500, 500), train_scale = (0.5, 1.5), mean=(128, 128, 128), std = (1,1,1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.pseudo_root = pseudo_root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.crop_h, self.crop_w = crop_size
        self.lscale, self.hscale = train_scale
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            if self.pseudo_root == None:
                label_file = osp.join(self.root, label_name)
            else:
                label_file = label_name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):

        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = self.lscale + random.randint(0, int((self.hscale-self.lscale)*10)) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # OpenCV read image as BGR, not RGB
        label = np.array(Image.open(datafiles["label"]))
        #
        sys.path.insert(0, 'dataset/helpers')
        from labels_synthia import id2label, trainId2label
        #
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        # id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        # valid_labels = sorted(set(id_2_label.ravel()))
        label = label_2_id[label]
        #
        size = image.shape
        img_name = datafiles["img_name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = image/255.0 # scale to [0,1]
        image -= self.mean # BGR
        image = image/self.std#np.reshape(self.std,(1,1,3))

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), img_name

class SYNTHIASTDataSet(data.Dataset):
    def __init__(self, root, list_path, reg_weight = 0.0, pseudo_root = None, max_iters=None, crop_size=(500, 500), train_scale = (0.5, 1.5), mean=(128, 128, 128), std = (1,1,1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.pseudo_root = pseudo_root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.crop_h, self.crop_w = crop_size
        self.lscale, self.hscale = train_scale
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        self.reg_weight = reg_weight
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            if self.pseudo_root == None:
                label_file = osp.join(self.root, label_name)
            else:
                label_file = label_name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):

        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = self.lscale + random.randint(0, int((self.hscale-self.lscale)*10)) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # OpenCV read image as BGR, not RGB
        label = np.array(Image.open(datafiles["label"]))
        #
        sys.path.insert(0, 'dataset/helpers')
        from labels_synthia import id2label, trainId2label
        #
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        # id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        # valid_labels = sorted(set(id_2_label.ravel()))
        label = label_2_id[label]
        #
        size = image.shape
        img_name = datafiles["img_name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = image/255.0 # scale to [0,1]
        image -= self.mean # BGR
        image = image/self.std#np.reshape(self.std,(1,1,3))

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), img_name, self.reg_weight

class GTA5STDataSet(data.Dataset):
    def __init__(self, root, list_path, reg_weight = 0.0, pseudo_root = None, max_iters=None, crop_size=(500, 500), train_scale = (0.5, 1.5), mean=(128, 128, 128), std = (1,1,1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.pseudo_root = pseudo_root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.crop_h, self.crop_w = crop_size
        self.lscale, self.hscale = train_scale
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        self.reg_weight = reg_weight
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            if self.pseudo_root == None:
                label_file = osp.join(self.root, label_name)
            else:
                label_file = label_name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):

        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = self.lscale + random.randint(0, int((self.hscale-self.lscale)*10)) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # OpenCV read image as BGR, not RGB
        label = np.array(Image.open(datafiles["label"]))
        #
        sys.path.insert(0, 'dataset/helpers')
        from labels import id2label, trainId2label
        #
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        # id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        # valid_labels = sorted(set(id_2_label.ravel()))
        label = label_2_id[label]
        #
        size = image.shape
        img_name = datafiles["img_name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = image/255.0 # scale to [0,1]
        image -= self.mean # BGR
        image = image/self.std#np.reshape(self.std,(1,1,3))

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), img_name, self.reg_weight

class SrcSTDataSet(data.Dataset):
    def __init__(self, root, list_path, data_src=None, reg_weight = 0.0, pseudo_root = None, max_iters=None, crop_size=(500, 500), train_scale = (0.5, 1.5), mean=(128, 128, 128), std = (1,1,1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.pseudo_root = pseudo_root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.crop_h, self.crop_w = crop_size
        self.lscale, self.hscale = train_scale
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        self.reg_weight = reg_weight
        self.data_src = data_src
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            if self.pseudo_root == None:
                label_file = osp.join(self.root, label_name)
            else:
                label_file = label_name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):

        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = self.lscale + random.randint(0, int((self.hscale-self.lscale)*10)) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # OpenCV read image as BGR, not RGB
        label = np.array(Image.open(datafiles["label"]))
        #
        sys.path.insert(0, 'dataset/helpers')
        if self.data_src == 'gta':
            from labels import id2label
        elif self.data_src == 'synthia':
            from labels_synthia import id2label
        #
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        # id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        # valid_labels = sorted(set(id_2_label.ravel()))
        label = label_2_id[label]
        #
        size = image.shape
        img_name = datafiles["img_name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = image/255.0 # scale to [0,1]
        image -= self.mean # BGR
        image = image/self.std#np.reshape(self.std,(1,1,3))

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), img_name, self.reg_weight

class SoftSrcSTDataSet(data.Dataset):
    def __init__(self, root, list_path, data_src = None, num_classes = None, reg_weight = 0.0, pseudo_root = None, max_iters=None, crop_size=(500, 500), train_scale = (0.5, 1.5), mean=(128, 128, 128), std = (1,1,1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.pseudo_root = pseudo_root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.crop_h, self.crop_w = crop_size
        self.lscale, self.hscale = train_scale
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        self.reg_weight = reg_weight
        self.data_src = data_src
        self.num_classes = num_classes
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            if self.pseudo_root == None:
                label_file = osp.join(self.root, label_name)
            else:
                label_file = label_name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):

        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = self.lscale + random.randint(0, int((self.hscale-self.lscale)*10)) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # OpenCV read image as BGR, not RGB
        label = np.array(Image.open(datafiles["label"]))
        #
        sys.path.insert(0, 'dataset/helpers')
        if self.data_src == 'gta':
            from labels import id2label
        elif self.data_src == 'synthia':
            from labels_synthia import id2label
        #
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        # id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        # valid_labels = sorted(set(id_2_label.ravel()))
        label = label_2_id[label]
        #
        size = image.shape
        img_name = datafiles["img_name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = image/255.0 # scale to [0,1]
        image -= self.mean # BGR
        image = image/self.std#np.reshape(self.std,(1,1,3))

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        label_expand = np.tile( np.expand_dims(label, axis=2), (1, 1, self.num_classes) )
        labels = label_expand.copy()
        labels[labels != self.ignore_label] = 1.0
        labels[labels == self.ignore_label] = 0.0
        labels = np.cumsum(labels, axis=2)
        labels[labels != label_expand + 1] = 0.0
        del label_expand
        labels[labels != 0.0] = 1.0
        labels = labels.transpose((2,0,1))

        # weighted_pred_trainIDs = np.asarray(np.argmax(labels, axis=0), dtype=np.uint8)
        # # save weighted predication
        # wpred_label_col = weighted_pred_trainIDs.copy()
        # wpred_label_col = colorize_mask(wpred_label_col)
        # wpred_label_col.save('%s_color.png' % (index))
        #
        # labels_sum = np.sum(labels,axis=0)
        # # save weighted predication
        # weighted_pred_trainIDs[labels_sum == 0] = 255
        # wpred_label_col = weighted_pred_trainIDs.copy()
        # wpred_label_col = colorize_mask(wpred_label_col)
        # wpred_label_col.save('%s_pseudo_color.png' % (index))

        return image.copy(), labels.copy(), np.array(size), img_name, self.reg_weight

class SoftGTA5StMineDataSet(data.Dataset):
    def __init__(self, root, list_path, data_src=None, num_classes = None, reg_weight = 0.0, rare_id = None, mine_id = None, mine_chance = None, pseudo_root = None, max_iters=None, crop_size=(500, 500), train_scale = (0.5, 1.5), mean=(128, 128, 128), std = (1,1,1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.pseudo_root = pseudo_root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.crop_h, self.crop_w = crop_size
        self.lscale, self.hscale = train_scale
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        self.softlabel_ids = []
        self.reg_weight = reg_weight
        self.rare_id = rare_id
        self.mine_id = mine_id
        self.mine_chance = mine_chance
        self.data_src = data_src
        self.num_classes = num_classes
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
                self.softlabel_ids.append(fields[2])
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
            self.softlabel_ids = self.softlabel_ids * int(np.ceil(float(max_iters) / len(self.softlabel_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            softlabel_name = self.softlabel_ids[idx]
            img_file = osp.join(self.root, img_name)
            if self.pseudo_root == None:
                label_file = osp.join(self.root, label_name)
                softlabel_file = osp.join(self.root, softlabel_name)
            else:
                label_file = label_name
                softlabel_file = softlabel_name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "softlabel": softlabel_file,
                "img_name": img_name,
                "label_name": label_name,
                "softlabel_name": softlabel_name
            })

    def __len__(self):

        return len(self.files)

    def generate_scale_label(self, image, label, input_softlabel):
        # f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = self.lscale + random.randint(0, int((self.hscale-self.lscale)*10)) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        # interpolate the softlabel by 3-channel groups
        h,w = label.shape
        num_group = int(np.ceil(self.num_classes/3.0))
        softlabel = np.zeros((h,w,self.num_classes), dtype=np.float32)
        start_idx = 0
        for idx in range(num_group):
            clabel = input_softlabel[:,:,start_idx:start_idx+3]
            clabel = cv2.resize(clabel, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            softlabel[:,:,start_idx:start_idx+3] = clabel.reshape(h,w,-1)
            start_idx = start_idx + 3
        softlabel = softlabel.transpose(2,0,1)/np.sum(softlabel,2)
        softlabel = softlabel.transpose(1,2,0)
        return image, label, softlabel

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # OpenCV read image as BGR, not RGB
        label = np.array(Image.open(datafiles["label"]))
        softlabel = np.load(datafiles["softlabel"])
        #
        sys.path.insert(0, 'dataset/helpers')
        if self.data_src == 'gta':
            from labels import id2label
        elif self.data_src == 'synthia':
            from labels_cityscapes_synthia import id2label
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        label = label_2_id[label]
        #
        size = image.shape
        img_name = datafiles["img_name"]
        if self.scale:
            image, label, softlabel = self.generate_scale_label(image, label, softlabel)
        image = np.asarray(image, np.float32)
        image = image/255.0 # scale to [0,1]
        image -= self.mean # BGR
        image = image/self.std # np.reshape(self.std,(1,1,3))

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
            # softlabel_pad
            h_pad, w_pad = label_pad.shape
            num_group = int(np.ceil(self.num_classes / 3.0))
            softlabel_pad = np.zeros((h_pad, w_pad, self.num_classes), dtype=np.float32)
            start_idx = 0
            for idx in range(num_group):
                clabel_pad = softlabel[:, :, start_idx:start_idx + 3]
                clabel_pad = cv2.copyMakeBorder(clabel_pad, 0, pad_h, 0,pad_w, cv2.BORDER_CONSTANT,value=(0.0, 0.0, 0.0))
                softlabel_pad[:, :, start_idx:start_idx + 3] = clabel_pad.reshape(h_pad,w_pad,-1)
                start_idx = start_idx + 3
        else:
            img_pad, label_pad, softlabel_pad = image, label, softlabel

        img_h, img_w = label_pad.shape
        # mining or not
        mine_flag = random.uniform(0, 1) < self.mine_chance
        if mine_flag and len(self.mine_id) > 0:
            label_unique = np.unique(label_pad)
            mine_id_temp = np.array([a for a in self.mine_id if a in label_unique]) # if this image has the mine id
            if mine_id_temp.size != 0:
                # decide the single id to be mined
                mine_id_img = mine_id_temp
                sel_idx = random.randint(0, mine_id_temp.size-1)
                sel_mine_id = mine_id_img[sel_idx]
                # seed for the mined id
                mine_id_loc = np.where(label_pad == sel_mine_id)  # tuple
                mine_id_len = len(mine_id_loc[0])
                seed_loc = random.randint(0, mine_id_len-1)
                hseed = mine_id_loc[0][seed_loc]
                wseed = mine_id_loc[1][seed_loc]
                # patch crop
                half_crop_h = self.crop_h/2
                half_crop_w = self.crop_w/2
                # center crop at the seed
                left_idx = wseed - half_crop_w
                right_idx = wseed + half_crop_w -1
                up_idx = hseed - half_crop_h
                bottom_idx = hseed + half_crop_h - 1
                # shift the left_idx or right_idx if they go beyond the pad margins
                if left_idx < 0:
                    left_idx = 0
                elif right_idx > img_w - 1:
                    left_idx = left_idx - ( ( half_crop_w - 1 ) - (img_w - 1 - wseed) ) # left_idx shifts to the left by the right beyond length
                if up_idx < 0:
                    up_idx = 0
                elif bottom_idx > img_h - 1:
                    up_idx = up_idx - ( ( half_crop_h - 1 ) - (img_h - 1 - hseed) ) # up_idx shifts to the up by the bottom beyond length
                h_off = up_idx
                w_off = left_idx
            else:
                h_off = random.randint(0, img_h - self.crop_h)
                w_off = random.randint(0, img_w - self.crop_w)
        else:
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        softlabel = np.asarray(softlabel_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))
        # set ignored label vector to be all zeros
        label_expand = np.tile( np.expand_dims(label, axis=2), (1, 1, self.num_classes) )
        labels_ = label_expand.copy()
        labels_[labels_ != self.ignore_label] = 1.0
        labels_[labels_ == self.ignore_label] = 0.0
        labels = labels_*softlabel
        labels = labels.transpose((2,0,1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            labels = labels[:,:,::flip]

        # weighted_pred_trainIDs = np.asarray(np.argmax(labels, axis=0), dtype=np.uint8)
        # # save weighted predication
        # wpred_label_col = weighted_pred_trainIDs.copy()
        # wpred_label_col = colorize_mask(wpred_label_col)
        # wpred_label_col.save('%s_color.png' % (index))
        #
        # labels_sum = np.sum(labels,axis=0)
        # # save weighted predication
        # weighted_pred_trainIDs[labels_sum == 0] = 255
        # wpred_label_col = weighted_pred_trainIDs.copy()
        # wpred_label_col = colorize_mask(wpred_label_col)
        # wpred_label_col.save('%s_pseudo_color.png' % (index))

        return image.copy(), labels.copy(), np.array(size), img_name, self.reg_weight



class GTA5StMineDataSet(data.Dataset):
    def __init__(self, root, list_path, data_src=None, reg_weight = 0.0, rare_id = None, mine_id = None, mine_chance = None, pseudo_root = None, max_iters=None, crop_size=(500, 500), train_scale = (0.5, 1.5), mean=(128, 128, 128), std = (1,1,1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.pseudo_root = pseudo_root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.crop_h, self.crop_w = crop_size
        self.lscale, self.hscale = train_scale
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        self.reg_weight = reg_weight
        self.rare_id = rare_id
        self.mine_id = mine_id
        self.mine_chance = mine_chance
        self.data_src = data_src
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            if self.pseudo_root == None:
                label_file = osp.join(self.root, label_name)
            else:
                label_file = label_name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):

        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = self.lscale + random.randint(0, int((self.hscale-self.lscale)*10)) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # OpenCV read image as BGR, not RGB
        label = np.array(Image.open(datafiles["label"]))
        #
        sys.path.insert(0, 'dataset/helpers')
        if self.data_src == 'gta':
            from labels import id2label
        elif self.data_src == 'synthia':
            from labels_cityscapes_synthia import id2label
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        label = label_2_id[label]
        #
        size = image.shape
        img_name = datafiles["img_name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = image/255.0 # scale to [0,1]
        image -= self.mean # BGR
        image = image/self.std # np.reshape(self.std,(1,1,3))

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        # mining or not
        mine_flag = random.uniform(0, 1) < self.mine_chance
        if mine_flag and len(self.mine_id) > 0:
            label_unique = np.unique(label_pad)
            mine_id_temp = np.array([a for a in self.mine_id if a in label_unique]) # if this image has the mine id
            if mine_id_temp.size != 0:
                # decide the single id to be mined
                mine_id_img = mine_id_temp
                sel_idx = random.randint(0, mine_id_temp.size-1)
                sel_mine_id = mine_id_img[sel_idx]
                # seed for the mined id
                mine_id_loc = np.where(label_pad == sel_mine_id)  # tuple
                mine_id_len = len(mine_id_loc[0])
                seed_loc = random.randint(0, mine_id_len-1)
                hseed = mine_id_loc[0][seed_loc]
                wseed = mine_id_loc[1][seed_loc]
                # patch crop
                half_crop_h = self.crop_h/2
                half_crop_w = self.crop_w/2
                # center crop at the seed
                left_idx = wseed - half_crop_w
                right_idx = wseed + half_crop_w -1
                up_idx = hseed - half_crop_h
                bottom_idx = hseed + half_crop_h - 1
                # shift the left_idx or right_idx if they go beyond the pad margins
                if left_idx < 0:
                    left_idx = 0
                elif right_idx > img_w - 1:
                    left_idx = left_idx - ( ( half_crop_w - 1 ) - (img_w - 1 - wseed) ) # left_idx shifts to the left by the right beyond length
                if up_idx < 0:
                    up_idx = 0
                elif bottom_idx > img_h - 1:
                    up_idx = up_idx - ( ( half_crop_h - 1 ) - (img_h - 1 - hseed) ) # up_idx shifts to the up by the bottom beyond length
                h_off = up_idx
                w_off = left_idx
            else:
                h_off = random.randint(0, img_h - self.crop_h)
                w_off = random.randint(0, img_w - self.crop_w)
        else:
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), img_name, self.reg_weight

class GTA5TestDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, test_size=(1024, 512), test_scale = 1.0, mean=(128, 128, 128), std = (1,1,1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.test_h, self.test_w = test_size
        self.scale = scale
        self.test_scale = test_scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            label_file = osp.join(self.root, label_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # OpenCV read image as BGR, not RGB
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        #
        # sys.path.insert(0, 'dataset/helpers')
        # from labels import id2label, trainId2label
        # #
        # label_2_id = 255 * np.ones((256,))
        # for l in id2label:
        #     if l in (-1, 255):
        #         continue
        #     label_2_id[l] = id2label[l].trainId
        # # id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        # # valid_labels = sorted(set(id_2_label.ravel()))
        # label = label_2_id[label]
        #
        # resize
        img_name = datafiles["img_name"]
        # image = cv2.resize(image, (self.test_h, self.test_w), fx=0, fy=0, interpolation = cv2.INTER_LINEAR)
        # label = cv2.resize(label, (self.test_h, self.test_w), fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        image = cv2.resize(image, None, fx=self.test_scale, fy=self.test_scale, interpolation = cv2.INTER_LINEAR)
        # always keep the resolution of label unchanged
        # label = cv2.resize(label, None, fx=1, fy=1, interpolation = cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        image = image/255.0 # scale to [0,1]
        image -= self.mean # BGR
        image = image/self.std#np.reshape(self.std,(1,1,3))
        size = image.shape
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), img_name

class GTA5MSTDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, test_size=(1024, 512), test_scale = 1.0, mean=(128, 128, 128), std = (1,1,1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.test_h, self.test_w = test_size
        self.scale = scale
        self.test_scale = test_scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            label_file = osp.join(self.root, label_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # OpenCV read image as BGR, not RGB
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        #
        # sys.path.insert(0, 'dataset/helpers')
        # from labels import id2label, trainId2label
        # #
        # label_2_id = 255 * np.ones((256,))
        # for l in id2label:
        #     if l in (-1, 255):
        #         continue
        #     label_2_id[l] = id2label[l].trainId
        # # id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        # # valid_labels = sorted(set(id_2_label.ravel()))
        # label = label_2_id[label]
        #
        # resize
        img_name = datafiles["img_name"]
        # image = cv2.resize(image, (self.test_h, self.test_w), fx=0, fy=0, interpolation = cv2.INTER_LINEAR)
        # label = cv2.resize(label, (self.test_h, self.test_w), fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        image = cv2.resize(image, None, fx=self.test_scale, fy=self.test_scale, interpolation = cv2.INTER_LINEAR)
        # always keep the resolution of label unchanged
        # label = cv2.resize(label, None, fx=1, fy=1, interpolation = cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        # image = image/255.0 # scale to [0,1]
        # image -= self.mean # BGR
        # image = image/self.std#np.reshape(self.std,(1,1,3))
        size = image.shape
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), img_name

class GTA5TestCRFDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, test_size=(1024, 512), test_scale = 1.0, mean=(128, 128, 128), std = (1,1,1), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.test_h, self.test_w = test_size
        self.scale = scale
        self.test_scale = test_scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = []
        self.label_ids = []
        with open(list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)
            label_file = osp.join(self.root, label_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # OpenCV read image as BGR, not RGB
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        #
        # sys.path.insert(0, 'dataset/helpers')
        # from labels import id2label, trainId2label
        # #
        # label_2_id = 255 * np.ones((256,))
        # for l in id2label:
        #     if l in (-1, 255):
        #         continue
        #     label_2_id[l] = id2label[l].trainId
        # # id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        # # valid_labels = sorted(set(id_2_label.ravel()))
        # label = label_2_id[label]
        #
        # resize
        img_name = datafiles["img_name"]
        # image = cv2.resize(image, (self.test_h, self.test_w), fx=0, fy=0, interpolation = cv2.INTER_LINEAR)
        # label = cv2.resize(label, (self.test_h, self.test_w), fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        image_crf = np.asarray(image, np.float32)
        image_crf = image_crf[:, :, ::-1]  # change to RGB
        image = cv2.resize(image, None, fx=self.test_scale, fy=self.test_scale, interpolation = cv2.INTER_LINEAR)
        # always keep the resolution of label unchanged
        # label = cv2.resize(label, None, fx=1, fy=1, interpolation = cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        image = image/255.0 # scale to [0,1]
        image -= self.mean # BGR
        image = image/self.std#np.reshape(self.std,(1,1,3))
        size = image.shape
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), image_crf.copy(), np.array(size), img_name

class VOCDataTestSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(505, 505), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = [] 
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "img/%s.jpg" % name)
            self.files.append({
                "img": img_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean
        
        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
        image = image.transpose((2, 0, 1))
        return image, name, size


if __name__ == '__main__':
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
