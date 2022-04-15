import cv2
import os
import random
import numpy as np
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.degradations import add_jpg_compression
from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY

def make_cit_dataset(path):
    clips_path = []

    for index, folder in enumerate(os.listdir(path)):
        clips_folder = os.path.join(path, folder)
        if not (os.path.isdir(clips_folder)):
            continue
        clips_path.append([])

        for image in sorted(os.listdir(clips_folder)):
            clips_path[index].append(os.path.join(clips_folder, image))
    return clips_path

@DATASET_REGISTRY.register()
class CITDataset(data.Dataset):
    """Continuous image transition dataset.
    """

    def __init__(self, opt):
        super(CITDataset, self).__init__()
        self.opt = opt 
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task']
        self.extend_t = opt['extend_t'] if 'extend_t' in opt else False

        self.cit_folders = opt['dataroot_cit']

        self.paths = []
        for cit_folder in self.cit_folders:
            self.paths += make_cit_dataset(cit_folder)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        num_frame = len(self.paths[index])
        idxes = sorted(random.sample(range(num_frame), 3))
        if self.task == 'i2i' or self.task == 'morphing':
            idxes[0], idxes[2] = 0, num_frame - 1

        if self.opt['phase'] == 'train':
            if random.random() < 0.5:
                idxes.reverse()

            # optional tricks
            if random.random() < 0.25 and self.task != 'vfi':
                idxes[1] = num_frame-1 if self.task == 'i2i' else (num_frame-1)//2

        t = np.float32((idxes[1] - idxes[0]) / (idxes[2] - idxes[0])) # float32

        # extend t from [0, 1] to [-1, 1]
        if self.extend_t and random.random() < 0.5:
            idxes = [num_frame-1-idx for idx in idxes]
            t = -t

        frame_paths = [self.paths[index][idx] for idx in idxes]

        img_bytes = self.file_client.get(frame_paths[1], 'frame_gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(frame_paths[0], 'frame_0')
        img_0 = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(frame_paths[2], 'frame_1')
        img_1 = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            if self.task == 'vfi':
                # random crop
                img_gt, img_in = paired_random_crop(img_gt, [img_0, img_1], gt_size, 1)
                img_0, img_1 = img_in
            else:
                img_gt = cv2.resize(img_gt, (gt_size, gt_size))
                img_0 = cv2.resize(img_0, (gt_size, gt_size))
                img_1 = cv2.resize(img_1, (gt_size, gt_size))
            # flip, rotation
            img_gt, img_0, img_1 = augment([img_gt, img_0, img_1], self.opt['use_flip'], self.opt['use_rot'])
        else:
            h, w = img_gt.shape[:2]
            nh, nw = h//32*32, w//32*32
            img_gt, img_0, img_1 = img_gt[:nh,:nw], img_0[:nh,:nw], img_1[:nh,:nw]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_0, img_1 = img2tensor([img_gt, img_0, img_1], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_0, self.mean, self.std, inplace=True)
            normalize(img_1, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        inputs = img_0 if self.task=='i2i' else torch.cat((img_0, img_1), 0)
        return {'in': inputs, 'gt': img_gt, 't': t, 'lq_path': frame_paths[0], 'gt_path': frame_paths[1]}

    def __len__(self):
        return len(self.paths)