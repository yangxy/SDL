import torch
from collections import OrderedDict
from os import path as osp 
from tqdm import tqdm
import numpy as np

from basicsr.archs import build_network
from losses import build_loss_
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

from torch.nn import functional as F
import torchvision.transforms.functional as TF

@MODEL_REGISTRY.register()
class SDLStyleModel(SRModel):
    """SDL model for video frame interpolation and beyond."""

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('contentstyle_opt'):
            self.cri_cs = build_loss_(train_opt['contentstyle_opt']).to(self.device)
        else:
            self.cri_cs = None

        if train_opt.get('contentstylerelt_opt'):
            self.cri_csr = build_loss_(train_opt['contentstylerelt_opt']).to(self.device)
        else:
            self.cri_csr = None

        if self.cri_cs is None and self.cri_csr is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.content = data['content'].to(self.device)
        self.style = data['style'].to(self.device)
        self.t = data['t'].to(self.device)
        
        self.input = torch.cat((self.content, self.style), dim=1)

        self.tv = 1.0
        if np.random.random() < 0.5:
            self.tv = np.random.random()
        self.t = self.t*0 + self.tv

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.input, self.t)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_cs:
            l_content, l_style = self.cri_cs(self.output, self.content, self.style, norm=True)
            l_cs = l_content + self.tv*l_style
            l_total += l_cs
            loss_dict['l_content'] = l_content
            loss_dict['l_style'] = l_style
        if self.cri_csr:
            l_content_r, l_style_r = self.cri_csr(self.output, self.content, self.style)
            l_csr = l_content_r + self.tv*l_style_r
            l_total += l_csr
            loss_dict['l_content_r'] = l_content_r
            loss_dict['l_style_r'] = l_style_r

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)