# Datasets

This folder is mainly for storing datasets used for training/validation/testing.

### VFI

Make data structure to be:
```
├────vimeo_triplet
	├────train
		├────00001_0001
			├────im1.png
			├────im2.png
			├────im3.png
		├────00001_0002
			├────im1.png
			├────im2.png
			├────im3.png
		...

	├────validation
		├────00001_0389
			├────im1.png
			├────im2.png
			├────im3.png
		...
```
- Training

	[Adobe240fps](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/) | [vimeo triplet](http://toflow.csail.mit.edu/)

- Testing

	[UCF101 triplet](https://github.com/liuziwei7/voxel-flow) | [Middleburry](https://vision.middlebury.edu/flow/data/)

Please following this [instruction](https://github.com/avinashpaliwal/Super-SloMo/tree/master/data) to prepare Adobe240fps training/validation sets.

### Morphing

- Training

	dog2dog | cat2cat

### I2I

- Training

	Face aging | Face toonification


### Style Transfer

- Training

	[coco](https://cocodataset.org/) | [wikiart](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)

### Super Resolution

- Training
 	
 	[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)