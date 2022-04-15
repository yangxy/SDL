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

@DATASET_REGISTRY.register()
class StyleDataset(data.Dataset):
    """Style transfer dataset.
    """

    def __init__(self, opt):
        super(StyleDataset, self).__init__()
        self.opt = opt 
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task']

        self.content_folders = opt['dataroot_content']
        self.style_folders = opt['dataroot_style']

        self.paths, self.paths_style = [], []
        for content_folder in self.content_folders:
            self.paths += sorted(list(scandir(content_folder, recursive=True, full_path=True)))

        for style_folder in self.style_folders:
            self.paths_style += sorted(list(scandir(style_folder, recursive=True, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        index_s = random.randint(0, len(self.paths_style)-1)

        t = np.float32(random.random()) # float32

        img_bytes = self.file_client.get(self.paths[index], 'content')
        img_content = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(self.paths_style[index_s], 'style')
        img_style = imfrombytes(img_bytes, float32=True)

        img_content = cv2.resize(img_content, (512, 512))
        img_style = cv2.resize(img_style, (512, 512))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_content, img_style = paired_random_crop(img_content, img_style, gt_size, 1)
            # flip, rotation
            img_content, img_style = augment([img_content, img_style], self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_content, img_style = img2tensor([img_content, img_style], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_content, self.mean, self.std, inplace=True)
            normalize(img_style, self.mean, self.std, inplace=True)

        return {'content': img_content, 'style': img_style, 't': t, 'content_path': self.paths[index], 'style_path': self.paths_style[index_s]}

    def __len__(self):
        return len(self.paths)