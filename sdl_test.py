import os
import math
import cv2 
import glob
import shutil
import random
import torch
import argparse
import numpy as np
from copy import deepcopy
from archs.sdl_arch import SDLNet
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.utils import FileClient, imfrombytes, img2tensor, tensor2img, imwrite

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

class SDLWrap(object):
    def __init__(self, model_path, split=0.5, in_ch=6, key=None):
        self.split = split
        self.model_path = model_path
        self.file_client = FileClient(backend='disk')
        
        self.load_model(in_ch, key)

    def load_model(self, in_ch, key=None):
        self.net = SDLNet(in_ch, 3, self.split, nrow=3, ncol=6)
        load_net = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        if key is not None: load_net = load_net[key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v 
                load_net.pop(k)
        self.net.load_state_dict(load_net, strict=True)
        self.net.cuda()
        self.net.eval()

    def validate_vfi(self, val_path, save_img=False):
        scenes = make_cit_dataset(val_path)
        psnr_sum = 0
        ssim_sum = 0
        num = 0
        for n, scene in enumerate(scenes[:]):
            if n%100==0: print(f'{n}/{len(scenes)}')
            l = len(scene)
            img_bytes = self.file_client.get(scene[0], 'frame0')
            img_0 = imfrombytes(img_bytes, float32=True)
            img_bytes = self.file_client.get(scene[-1], 'frame0')
            img_1 = imfrombytes(img_bytes, float32=True)

            h, w = img_0.shape[:2]
            hh, ww = math.ceil(h/32)*32, math.ceil(w/32)*32
            img_0 = cv2.copyMakeBorder(img_0, 0, hh-h, 0, ww-w, cv2.BORDER_REFLECT)
            img_1 = cv2.copyMakeBorder(img_1, 0, hh-h, 0, ww-w, cv2.BORDER_REFLECT)

            img_0, img_1 = img2tensor([img_0, img_1], bgr2rgb=True, float32=True)
            img_01 = torch.cat((img_0, img_1), 0).unsqueeze(0).cuda()

            #for idx in range(0, l):
            for idx in range(1, l-1):
                img_bytes = self.file_client.get(scene[idx], 'framet')
                img_gt = imfrombytes(img_bytes, float32=True)
                img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True).unsqueeze(0).cuda()
                img_gt = tensor2img(img_gt)

                with torch.no_grad():
                    t = torch.FloatTensor([idx/(l-1)]).unsqueeze(0).cuda()
                    img_t = self.net(img_01, t)
                img_t = tensor2img(img_t)
                img_t = img_t[:h, :w]
                if save_img:
                    save_path = os.path.dirname(scene[0])
                    imwrite(img_t, os.path.join(save_path, f'{idx:02d}_sdl.png'))
                psnr = calculate_psnr(img_t, img_gt, 0)
                ssim = calculate_ssim(img_t, img_gt, 0)
                psnr_sum += psnr
                ssim_sum += ssim
                num += 1
                #print(psnr)
        print('total:', psnr_sum/num, ssim_sum/num)

    def test_vfi(self, source, target, save_path, num=1):
        img_bytes = self.file_client.get(source, 'frame0')
        img_0 = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(target, 'frame1')
        img_1 = imfrombytes(img_bytes, float32=True)

        h, w = img_0.shape[:2]
        hh, ww = math.ceil(h/32)*32, math.ceil(w/32)*32
        img_0 = cv2.copyMakeBorder(img_0, 0, hh-h, 0, ww-w, cv2.BORDER_REFLECT)
        img_1 = cv2.copyMakeBorder(img_1, 0, hh-h, 0, ww-w, cv2.BORDER_REFLECT)

        img_0, img_1 = img2tensor([img_0, img_1], bgr2rgb=True, float32=True)
        img_01 = torch.cat((img_0, img_1), 0).unsqueeze(0).cuda()
    
        for j in range(1, num+1):
            with torch.no_grad():
                t = torch.FloatTensor([j/(num+1)]).unsqueeze(0).cuda()
                img_t = self.net(img_01, t)
            img_t = tensor2img(img_t)
            imwrite(img_t[:h, :w], os.path.join(save_path, f'{j:02d}_sdl.png'))

    def test_vfi_dir(self, in_path, save_path, num=1, copy_flag=True, ext='.png'):
        os.makedirs(save_path, exist_ok=True)
        files = sorted(glob.glob(os.path.join(in_path, '*' + ext)))
        num_frame = len(files)

        cur = 0 
        for idx in range(num_frame-1):
            if idx%10==0: print(files[idx])
            if copy_flag:
                shutil.copyfile(files[idx], f'{save_path}/{cur:03d}{ext}')
                cur += 1

            img_bytes = self.file_client.get(files[idx], 'frame0')
            img_0 = imfrombytes(img_bytes, float32=True)
            img_bytes = self.file_client.get(files[idx+1], 'frame1')
            img_1 = imfrombytes(img_bytes, float32=True)
    
            h, w = img_0.shape[:2]
            hh, ww = math.ceil(h/32)*32, math.ceil(w/32)*32
            img_0 = cv2.copyMakeBorder(img_0, 0, hh-h, 0, ww-w, cv2.BORDER_REFLECT)
            img_1 = cv2.copyMakeBorder(img_1, 0, hh-h, 0, ww-w, cv2.BORDER_REFLECT)

            img_0, img_1 = img2tensor([img_0, img_1], bgr2rgb=True, float32=True)
            img_01 = torch.cat((img_0, img_1), 0).unsqueeze(0).cuda()

            start = 1 if copy_flag else 0
            end = num if (idx != num_frame-2 or copy_flag) else num+1
            for i in range(start, end+1):
                with torch.no_grad():
                    t = torch.FloatTensor([i/(num+1)]).unsqueeze(0).cuda()
                    img_t = self.net(img_01, t)
                img_t = tensor2img(img_t)
                imwrite(img_t[:h, :w] , f'{save_path}/{cur:03d}{ext}')
                cur += 1

        if copy_flag:
            shutil.copyfile(files[-1], f'{save_path}/{cur:03d}{ext}')

    def test_morphing(self, source, target, save_path, size=512, num=7):
        img_bytes = self.file_client.get(source, 'frame0')
        img_0 = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(target, 'frame1')
        img_1 = imfrombytes(img_bytes, float32=True)

        size = max(32, min(512, size)) # size: {32, 64, ..., 512}
        size = size//32*32
       
        img_0 = cv2.resize(img_0, (size, size))
        img_1 = cv2.resize(img_1, (size, size))

        img_0, img_1 = img2tensor([img_0, img_1], bgr2rgb=True, float32=True)
        img_01 = torch.cat((img_0, img_1), 0).unsqueeze(0).cuda()
    
        for j in range(1, num+1):
            with torch.no_grad():
                t = torch.FloatTensor([j/(num+1)]).unsqueeze(0).cuda()
                img_t = self.net(img_01, t)
            img_t = tensor2img(img_t)
            imwrite(img_t, os.path.join(save_path, f'{j:02d}_sdl.png'))

    def test_i2i(self, source, save_path, size=512, num=7, extend_t=False):
        img_bytes = self.file_client.get(source, 'frame0')
        img_0 = imfrombytes(img_bytes, float32=True)

        size = max(32, min(512, size)) # size: {32, 64, ..., 512}
        size = size//32*32

        img_0 = cv2.resize(img_0, (size, size))
            
        img_0 = img2tensor(img_0, bgr2rgb=True, float32=True)
        img_0 = img_0.unsqueeze(0).cuda()
        
        start = -num if extend_t else 0
        for j in range(start, num+1):
            with torch.no_grad():
                t = torch.FloatTensor([j/(num+1)]).unsqueeze(0).cuda()
                img_t = self.net(img_0, t)
            img_t = tensor2img(img_t)
            imwrite(img_t, os.path.join(save_path, f'{j-start:02d}_sdl.png'))

    def test_style_transfer(self, content, style, save_path, num=7):
        img_bytes = self.file_client.get(content, 'frame0')
        img_0 = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(style, 'frame1')
        img_1 = imfrombytes(img_bytes, float32=True)
       
        img_1 = cv2.resize(img_1, img_0.shape[:2][::-1])

        h, w = img_0.shape[:2]
        hh, ww = h//32*32, w//32*32
        img_0, img_1 = img_0[:hh, :ww], img_1[:hh, :ww]

        img_0, img_1 = img2tensor([img_0, img_1], bgr2rgb=True, float32=True)
        img_01 = torch.cat((img_0, img_1), 0).unsqueeze(0).cuda()
    
        for j in range(1, num+1):
            with torch.no_grad():
                t = torch.FloatTensor([j/(num+1)]).unsqueeze(0).cuda()
                img_t = self.net(img_01, t)
            img_t = tensor2img(img_t)
            imwrite(img_t, os.path.join(save_path, f'{j:02d}_sdl.png'))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vfi', help='task of SDL model')
    parser.add_argument('--model', type=str, default='SDL_vfi_perceptual.pth', help='SDL model')
    parser.add_argument('--key', type=str, default='params', help='key of SDL model')
    parser.add_argument('--split', type=float, default=0.5, help='SDL split ratio')
    parser.add_argument('--in_ch', type=int, default=6, help='input channel')
    parser.add_argument('--num', type=int, default=7, help='output number')
    parser.add_argument('--size', type=int, default=512, help='input size')
    parser.add_argument('--extend_t', action='store_true', help='extend t or not')
    parser.add_argument('--source', type=str, default=None, help='source/content input')
    parser.add_argument('--target', type=str, default=None, help='target/style input')
    parser.add_argument('--indir', type=str, default=None, help='output folder')
    parser.add_argument('--outdir', type=str, default='results', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model_path = os.path.join('weights', args.model)
    model = SDLWrap(model_path, args.split, args.in_ch, args.key)

    if args.task=='vfi':
        #model_path = 'weights/SDL_vfi_psnr.pth'
        #model_path = 'weights/SDL_vfi_perceptual.pth'

        #val_path = 'datasets/VFI/adobe240fps/validation'
        #val_path = 'datasets/VFI/ucf101_triplet/validation'
        #val_path = 'datasets/VFI/middleburry_other'
        #val_path = 'datasets/VFI/vimeo_triplet/validation'

        #model.validate_vfi(val_path, save_img=False)

        model.test_vfi(args.source, args.target, args.outdir, args.num)
    elif args.task=='vfi-dir':
        #model_path = 'weights/SDL_vfi_psnr.pth'
        #model_path = 'weights/SDL_vfi_perceptual.pth'

        model.test_vfi_dir(args.indir, args.outdir, args.num)
    elif args.task=='morphing':
        #model_path = 'weights/SDL_cat2cat_scale.pth'
        #model_path = 'weights/SDL_dog2dog_scale.pth'
        
        model.test_morphing(args.source, args.target, args.outdir, args.size, args.num)
    elif args.task=='i2i':
        #model_path = 'weights/SDL_aging_scale.pth'
        #model_path = 'weights/SDL_toonification_scale.pth'

        model.test_i2i(args.source, args.outdir, args.size, args.num, args.extend_t)
    elif args.task=='style_transfer':
        #model_path = 'weights/SDL_style_transfer_arbitrary.pth'

        model.test_style_transfer(args.source, args.target, args.outdir, args.num)

