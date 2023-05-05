# Image Stylization based on Deep Convolutional Neural Network



## Description

The project aims to design a deep convolutional neural network via transfer learning to apply special art effects to base/content image with style/reference image. The baseline model applies the pre-trained VGG-16 network. The style and content of the image are separated by using the characteristics of retaining semantic information at the high level and learning texture and other simple information at the bottom of the CNN network. Then, the input style image and content image are re-organized to form a stylized image. 

For futher improvements, we use the DualStyleGAN for comparison(Implementation details please refer https://github.com/williamyang1991/DualStyleGAN). Then, SSIM and PSNR are applied to evaluate the loss of content. In addition to this, we seek to use QT to design a GUI to accurately and quickly show the output image and this improves user experience.

## Getting Started

### Conda Environment Setup

To setup a conda environment with required dependencies, first install your favorite flavor of anaconda.
- For miniconda: [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- For full Anaconda: [Anaconda](https://www.anaconda.com/products/distribution)

### Configure environment.

Choose the `environment-XX.yml` file that best matches your system configuration.

- For computers with a CUDA capable GPU, this is `environment-CUDA.yml`
- For any Windows PC running windows 10 or later, this is `environment-WIN.yml`
- For anything not listed above (mac included), use `environment-CPU.yml`

then run the configuration to create a conda env:

```
conda env create -f YOUR_CONFIG.yml
```
The environment will be created according to the configuration chosen, and can then be activated with:

```
conda activate dl523
```

### Installing
Before starting our work, we have to setup our enivironment first and add all of the dependencies we need. They can be activated with:

```
pip install -r requirements.txt
```
## Help

## Authors

Zhenghao Sun    &nbsp; szh1007@bu.edu

Shangzhou Yin  &nbsp;  syin10@bu.edu

Guangrui Ding  &nbsp;  grding@bu.edu

Tom Panenko  &nbsp; tompan@bu.edu

## Version History

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Acknowledgments
