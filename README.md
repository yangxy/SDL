# Beyond a Video Frame Interpolator: A Space Decoupled Learning Approach to Continuous Image Transition

[Paper](https://arxiv.org/pdf/2203.09771) | Supplementary Material

[Tao Yang](https://cg.cs.tsinghua.edu.cn/people/~tyang)<sup>1</sup>, Peiran Ren<sup>1</sup>, Xuansong Xie<sup>1</sup>, Xiansheng Hua<sup>1</sup>, [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)<sup>2</sup>  
_<sup>1</sup>[DAMO Academy, Alibaba Group](https://damo.alibaba.com), Hangzhou, China_  
_<sup>2</sup>[Department of Computing, The Hong Kong Polytechnic University](http://www.comp.polyu.edu.hk), Hong Kong, China_

### Video Frame Interpolation

<img src="figs/vfi.png" width="784px"/>

### Face Aging

<img src="figs/aging.png" width="784px"/>

### Face Toonification

<img src="figs/toonification.png" width="784px"/>

### Image Morphing

<img src="figs/morphing.png" width="784px"/>

### Style Transfer

<img src="figs/style_transfer.png" width="784px"/>

## News
(2022-4-15) Add source codes and pre-trained models. Other pre-trained models will be released soon.

## Usage

![python](https://img.shields.io/badge/python-v3.7.4-green.svg?style=plastic)
![pytorch](https://img.shields.io/badge/pytorch-v1.7.0-green.svg?style=plastic)
![cuda](https://img.shields.io/badge/cuda-v10.2.89-green.svg?style=plastic)
![driver](https://img.shields.io/badge/driver-v460.73.01-green.svg?style=plastic)
![gcc](https://img.shields.io/badge/gcc-v7.5.0-green.svg?style=plastic)

- Clone this repository.
```bash
git clone https://github.com/yangxy/SDL.git
cd SDL
```

- Install dependencies. (Python 3 + NVIDIA GPU + CUDA. Recommend to use Anaconda)
```bash
pip install -r requirements.txt
````

- Download our pre-trained models and put them into ``weights/``. **Note:** SDL_vfi_psnr is obtained by finetuned SDL_vfi_perceptual using only Charbonnier Loss. It performs better in terms of PSNR but fails to recover fine details.

	[SDL_vfi_perceptual](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SDL/SDL_vfi_perceptual.pth) | [SDL_vfi_psnr](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SDL/SDL_vfi_psnr.pth) | [SDL_dog2dog_scale](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SDL/SDL_dog2dog_scale.pth) | [SDL_dog2dog_512](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SDL/SDL_dog2dog_512.pth) | [SDL_cat2cat_scale](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SDL/SDL_cat2cat_scale.pth) | [SDL_cat2cat_512](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SDL/SDL_cat2cat_512.pth) | [SDL_aging_scale](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SDL/SDL_aging_scale.pth) | [SDL_aging_512](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SDL/SDL_aging_512.pth) | [SDL_toonification_scale](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SDL/SDL_toonification_scale.pth) | [SDL_toonification_512](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SDL/SDL_toonification_512.pth) | [SDL_style_transfer_arbitrary](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/SDL/SDL_style_transfer_arbitrary.pth) | SDL_super_resolution (no plan at present)

- Test our models.
```bash
# Testing
python sdl_test.py --task vfi --model SDL_vfi_perceptual.pth --num 1 --source examples/VFI/car-turn/00000.jpg --target examples/VFI/car-turn/00001.jpg --outdir results/VFI

python sdl_test.py --task vfi-dir --model SDL_vfi_perceptual.pth --num 1 --indir examples/VFI/car-turn --outdir results/VFI

python sdl_test.py --task morphing --model SDL_dog2dog_scale.pth --num 7 --size 512 --source examples/Morphing/dog/source/flickr_dog_000045.jpg --target examples/Morphing/dog/target/pixabay_dog_000017.jpg --outdir results/Morphing/dog

python sdl_test.py --task morphing --model SDL_dog2dog_512.pth --num 7 --size 512 --source examples/Morphing/dog/source/flickr_dog_000045.jpg --target examples/Morphing/dog/target/pixabay_dog_000017.jpg --outdir results/Morphing/dog

python sdl_test.py --task morphing --model SDL_cat2cat_scale.pth --num 7 --size 512 --source examples/Morphing/cat/source/flickr_cat_000008.jpg --target examples/Morphing/cat/target/pixabay_cat_000010.jpg --outdir results/Morphing/cat

python sdl_test.py --task morphing --model SDL_cat2cat_512.pth --num 7 --size 512 --source examples/Morphing/cat/source/flickr_cat_000008.jpg --target examples/Morphing/cat/target/pixabay_cat_000010.jpg --outdir results/Morphing/cat

python sdl_test.py --task i2i --model SDL_aging_scale.pth --in_ch 3 --num 7 --size 512 --extend_t --source examples/I2I/ffhq-10/00002.png --outdir results/I2I/aging

python sdl_test.py --task i2i --model SDL_aging_512.pth --in_ch 3 --num 7 --size 512 --extend_t --source examples/I2I/ffhq-10/00002.png --outdir results/I2I/aging

python sdl_test.py --task i2i --model SDL_toonification_scale.pth --in_ch 3 --num 7 --size 512 --source examples/I2I/ffhq-10/00002.png --outdir results/I2I/toonification

python sdl_test.py --task i2i --model SDL_toonification_512.pth --in_ch 3 --num 7 --size 512 --source examples/I2I/ffhq-10/00002.png --outdir results/I2I/toonification

python sdl_test.py --task style_transfer --model SDL_style_transfer_arbitrary.pth --num 7 --source examples/Style_transfer/content/sailboat.jpg --target examples/Style_transfer/style/sketch.png --outdir results/Style_transfer
```

- Train SDL with 4 GPUs.
```bash
# Supervised training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_SDL_VFI.yml --auto_resume --launcher pytorch #--debug

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_SDLGAN_I2I.yml --auto_resume --launcher pytorch #--debug

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_SDLGAN_Morphing.yml --auto_resume --launcher pytorch #--debug

# Unsupervised training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_SDL_StyleTransfer.yml --auto_resume --launcher pytorch #--debug
```

- Prepare the training dataset by following this [instruction](datasets/README.md).

Please check out ``run.sh`` for more details.

## Citation
If our work is useful for your research, please consider citing:

    @inproceedings{Yang2022SDL,
	    title={Beyond a Video Frame Interpolator: A Space Decoupled Learning Approach to Continuous Image Transition},
	    author={Tao Yang, Peiran Ren, Xuansong Xie, Xiansheng Hua and Lei Zhang},
	    journal={European Conference on Computer Vision Workshop (ECCVW) (oral presentation)},
	    year={2022}
    }
    
## License
© Alibaba, 2022. For academic and non-commercial use only.

## Acknowledgments
This project is built based on the excellent [BasicSR-examples](https://github.com/xinntao/BasicSR-examples) project.

## Contact
If you have any questions or suggestions about this paper, feel free to reach me at yangtao9009@gmail.com.

