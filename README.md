# Image Stylization based on Deep Convolutional Neural Network

## Description

The project aims to design a deep convolutional neural network via transfer learning to apply special art effects to base/content image with style/reference image. The baseline model applies the pre-trained VGG-16 network. The style and content of the image are separated by using the characteristics of retaining semantic information at the high level and learning texture and other simple information at the bottom of the CNN network. Then, the input style image and content image are re-organized to form a stylized image. 

This technique has been widely used in many industrial applications, such as cartoons, videos, and Adobe Photoshop. We seek to apply artistic styles to portrait images with newly developed Deep convolutional neural networks. Previous studies introduce VGG-16 as a popular deep learning technique for image style transfer with the advantage of simplicity and fast training speed, so we implement it as the base model. 

However, the VGG-16 network cannot perform well for images with complex and detailed features such as portraits(i.e. wrinkles, hair and etc.), in order to keep more specific features for human faces, we use the DualStyleGAN for comparison (Implementation details please refer https://github.com/williamyang1991/DualStyleGAN). Then, SSIM and PSNR are applied to evaluate the loss of content. In addition to this, we seek to use QT to design a GUI to accurately and quickly show the output image and this improves user experience.

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

### Running VGG-16 network
Congratulations, When you are here, you are ready to run this code. choose the StyleTransfer-VGG16.py file, and then select your content image and style image. 

We have added sereval test images in the project folder. If you want to use your owen images, please remember replace the original image ```PATH``` with yours. 

```
content_img = imgLoad("TestPicture/image13.jpg")

style_img = imgLoad("TestPicture/image11.jpg")
```

After that, you can change the weight of content, weight of style, and num of epoch(All of this parameters can be changed by yourself):

```
content_weight = 1

style_weight = 1000

n_epoch = 200
```

Now, you can run this file and wait it patiently, then you will watch the result of this style transfer system.


### Running DualStyleGAN network



## Authors

Zhenghao Sun    &nbsp; szh1007@bu.edu

Shangzhou Yin  &nbsp;  syin10@bu.edu

Guangrui Ding  &nbsp;  grding@bu.edu

Tom Panenko  &nbsp; tompan@bu.edu

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


