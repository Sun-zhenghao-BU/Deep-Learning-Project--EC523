# Image Stylization based on Deep Convolutional Neural Network

## Description

The project aims to design a deep convolutional neural network via transfer learning to apply special art effects to base/content image with style/reference image. The baseline model applies the pre-trained VGG-16 network. The style and content of the image are separated by using the characteristics of retaining semantic information at the high level and learning texture and other simple information at the bottom of the CNN network. Then, the input style image and content image are re-organized to form a stylized image. 

This technique has been widely used in many industrial applications, such as cartoons, videos, and Adobe Photoshop. We seek to apply artistic styles to portrait images with newly developed Deep convolutional neural networks. Previous studies introduce VGG-16 as a popular deep learning technique for image style transfer with the advantage of simplicity and fast training speed, so we implement it as the base model. 

However, the VGG-16 network cannot perform well for images with complex and detailed features such as portraits(i.e. wrinkles, hair and etc.), in order to keep more specific features for human faces, we use the DualStyleGAN for comparison (The DualStyleGAN section refers from https://github.com/williamyang1991/DualStyleGAN with its implementation details). Then, SSIM and PSNR are applied to evaluate the loss of content. In addition to this, we seek to use QT to design a GUI to accurately and quickly show the output image and this improves user experience.

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

### Running DualStyleGAN network with pretrain model
Under Running DualStyleGAN folder
You can simply run DualStyleGAN.ipynb with GoogleColab

### Running DualStyleGAN network
This part is under DualStyleGAN folder

## Exemplar-Based Style Transfer
transfer the style of cartoon image into a face
```
python style_transfer.py
```

## Portrait Generation
Generate portraits images
```
python generate.py
```
You need to specify the style type with ```--style``` and the file name ```--namr```
```
python generate.py --style arcane --name arcane_generate
```
Next, you have to specify the weight to adjust the degree of style with ```--weight```, keep the intrinsic style code, extrinsic color code or extrinsic structure code fixed using --fix_content, --fix_color and --fix_structure.
```
python generate.py --style caricature --name caricature_generate --weight 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 --fix_content
```
## Training DualStyleGAN

Download the supporting models to the ./checkpoint/ folder:

Model	Description
stylegan2-ffhq-config-f.pt	StyleGAN model trained on FFHQ taken from rosinality.
model_ir_se50.pth	Pretrained IR-SE50 model taken from TreB1eN for ID loss.
## Facial Destylization
# Step 1: Prepare data. 
Prepare the dataset in ./data/DATASET_NAME/images/train/. First create lmdb datasets:

```
python ./model/stylegan/prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH
```

# Step 2: Fine-tune StyleGAN. Fine-tune StyleGAN in distributed settings:

```
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT finetune_stylegan.py --batch BATCH_SIZE \
       --ckpt FFHQ_MODEL_PATH --iter ITERATIONS --style DATASET_NAME --augment LMDB_PATH
```

# Step 3: Destylize artistic portraits.

```
python destylize.py --model_name FINETUNED_MODEL_NAME --batch BATCH_SIZE --iter ITERATIONS DATASET_NAME
```

The intrinsic and extrinsic style codes are saved in ```./checkpoint/cartoon/instyle_code.npy``` and ```./checkpoint/cartoon/exstyle_code.npy```, respectively. Intermediate results are saved in ```./log/cartoon/destylization/```. 

To speed up destylization, set ```--batch``` to large value such as 16. 

If the styles is very different from real faces, set ```--truncation``` to small value such as 0.5 to make the results more realistic (it enables DualStyleGAN to learn larger structrue deformations).

## Progressive Fine-Tuning
# Stage 1 & 2: Pretrain DualStyleGAN on FFHQ. This model is obtained by:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 pretrain_dualstylegan.py --iter 3000 --batch 4 ./data/ffhq/lmdb/
```
where ```./data/ffhq/lmdb/ contains the lmdb data created from the FFHQ dataset via ./model/stylegan/prepare_data.py```.

# Stage 3: Fine-Tune DualStyleGAN on Target Domain. Fine-tune DualStyleGAN in distributed settings:

```
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT finetune_dualstylegan.py --iter ITERATIONS \ 
                          --batch BATCH_SIZE --ckpt PRETRAINED_MODEL_PATH --augment DATASET_NAME
```
The loss term weights can be specified by ```--style_loss (λ_FM)```, ```--CX_loss (λ_CX)```, ```--perc_loss (λperc)```, ```--id_loss (λ_ID)``` and ```--L2_reg_loss (λreg)```. ```λID``` and ```λreg``` are suggested to be tuned for each style dataset to achieve ideal performance. More options can be found via python ```finetune_dualstylegan.py -h```.


The fine-tuned models can be found in ```./checkpoint/cartoon/generator-ITER.pt``` where ITER = 001000, 001100, ..., 001500. Intermediate results are saved in ```./log/cartoon/```. Large ITER has strong cartoon styles but at the cost of artifacts, and users may select the most balanced one from 1000-1500. We use 1400 as the same as teh original repository 


# Latent Optimization and Sampling
Refine extrinsic style code. Refine the color and structure styles to better fit the example style images.

```
python refine_exstyle.py --lr_color COLOR_LEARNING_RATE --lr_structure STRUCTURE_LEARNING_RATE DATASET_NAME
```

By default, the code will load ```instyle_code.npy```, ```exstyle_code.npy```, and ```generator.pt``` in ```./checkpoint/DATASET_NAME/```. Use ```--instyle_path```, ```--exstyle_path```, ```--ckpt``` to specify other saved style codes or models. 

The refined extrinsic style codes are saved in ```./checkpoint/DATASET_NAME/refined_exstyle_code.npy```. ```lr_color``` and ```lr_structure``` are suggested to be tuned to better fit the example styles.

Training sampling network. Train a sampling network to map unit Gaussian noises to the distribution of extrinsic style codes:

```python train_sampler.py DATASET_NAME```

By default, the code will load ```refined_exstyle_code.npy```or ```exstyle_code.npy``` in ```./checkpoint/DATASET_NAME/```. Use ```--exstyle_path``` to specify other saved extrinsic style codes. The saved model can be found in ```./checkpoint/DATASET_NAME/sampler.pt```.




## Authors

Zhenghao Sun    &nbsp; szh1007@bu.edu

Shangzhou Yin  &nbsp;  syin10@bu.edu

Guangrui Ding  &nbsp;  grding@bu.edu

Tom Panenko  &nbsp; tompan@bu.edu

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


