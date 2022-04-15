################### TEST #####################
#python sdl_test.py --task vfi --model SDL_vfi_perceptual.pth --num 1 --source examples/VFI/car-turn/00000.jpg --target examples/VFI/car-turn/00001.jpg --outdir results/VFI

python sdl_test.py --task vfi-dir --model SDL_vfi_perceptual.pth --num 1 --indir examples/VFI/car-turn --outdir results/VFI

#python sdl_test.py --task morphing --model SDL_dog2dog_scale.pth --num 7 --size 512 --source examples/Morphing/dog/source/flickr_dog_000045.jpg --target examples/Morphing/dog/target/pixabay_dog_000017.jpg --outdir results/Morphing/dog

#python sdl_test.py --task morphing --model SDL_cat2cat_scale.pth --num 7 --size 512 --source examples/Morphing/cat/source/flickr_cat_000008.jpg --target examples/Morphing/cat/target/pixabay_cat_000010.jpg --outdir results/Morphing/cat

#python sdl_test.py --task i2i --model SDL_aging_scale.pth --num 7 --size 512 --extend_t --source examples/I2I/ffhq-10/00002.png --outdir results/I2I/aging

#python sdl_test.py --task i2i --model SDL_toonification_scale.pth --num 7 --size 512 --source examples/I2I/ffhq-10/00002.png --outdir results/I2I/toonification

#python sdl_test.py --task style_transfer --model SDL_style_transfer_arbitrary.pth --num 7 --source examples/Style_transfer/content/sailboat.jpg --target examples/Style_transfer/style/sketch.png --outdir results/Style_transfer

################### TRAIN (supervised & unsupervised) #####################
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_SDL_VFI.yml --auto_resume --launcher pytorch #--debug

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_SDLGAN_I2I.yml --auto_resume --launcher pytorch #--debug

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_SDLGAN_Morphing.yml --auto_resume --launcher pytorch #--debug

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_SDL_StyleTransfer.yml --auto_resume --launcher pytorch #--debug