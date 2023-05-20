# High-resolution Imaging in Acoustic Microscopy using Deep Learning

This repo contains the code of all the models. The codes are basically cloned from other github repositories and changes were made either in the code or in the training data to incorporate our masking method.

## Abstract

Acoustic microscopy is a cutting-edge label-free imaging technology that allows to see the surface and interior structure of industrial and biological materials. The acoustic picture is created by focusing high-frequency acoustic waves on the object and then detecting reflected signals. The resolution of the acoustic image, on the other hand, is determined by the signal-to-noise ratio, scanning step size, and frequency of the transducer. Deep learning-based high-resolution imaging in acoustic microscopy is proposed in this paper. To illustrate 4 times resolution improvement in acoustic images, five distinct models are used: SRGAN, ESRGAN, IMDN, DBPN-RES-MR64-3, and SwinIR. The trained model’s performance is assessed by calculating the PSNR (Peak Signal to Noise Ratio) and SSIM (Structural Similarity Index) between the network-predicted and ground truth images. To avoid the model from over-fitting, transfer learning was incorporated during the procedure. SwinIR had average SSIM and PSNR values of 0.92 and 35, respectively. The model was also evaluated using a biological sample from Reindeer Antler, yielding an SSIM score of 0.8778 and a PSNR score of 32.93. Our framework is relevant to a wide range of industrial applications, including electronic production, material micro-structure analysis, and other biological applications in general.

## This figure depicts the overall strategy. First, acoustic data is acquired for both high (50 MHz) and low resolution (20 MHz). This is followed by deep learning-based high-resolution imaging. High-resolution ground truth and corresponding low-resolution input images are fed into the network used to train. The network output is further used as error backpropagation to train the network.

![image](https://github.com/banerjeepragyan/SuperResolution/assets/88557062/4a102326-d482-4c87-b93d-dc6c4a27d7c7)

## Diagram depicting the network architecture. The network consists of shallow feature extraction, deep feature extraction, and high-quality (HQ) image reconstruction modules. While the shallow feature extraction produces stable optimization and maps the input image space to a higher dimensional feature space, the model also has the deep feature extraction, consisting of K residual Swin transformer blocks (RSTB) and a 3 × 3 convolutional layer.

![image](https://github.com/banerjeepragyan/SuperResolution/assets/88557062/3b1e2425-eff6-4c78-96da-858ef6a433c5)

## Training

The training was initially done on the DIV2K dataset and then finetuned on the Acoustic dataset. 

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json
```

Here, in [options/swinir/train_swinir_sr_classical.json] refers to the json file where options are kept.

## Testing

The testing can be done by

```bash
python main_test_swinir.py --task classical_sr --scale 4 --training_patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth --folder_lq [LR_TestSet] --folder_gt [GT_TestSet]
```

Here, [LR_TestSet] and [GT_TestSet] refer to Ground Truth and Low Resolution Testsets.

## Results

The following results were obtained 

![image](https://github.com/banerjeepragyan/SuperResolution/assets/88557062/4a2b1755-f387-4a8f-88c7-1ccee9ea787b)

Deep learning was used to improve the lateral resolution of the SAM images. SRGAN, ESRGAN, IMDN, SwinIR, and DBPN-RES-MR64-3 were the models compared in this study. All five models were trained and tested on 17 different coins images, and the results were reported in terms of PSNR and SSIM scores. The SwinIR model is made up of modules for shallow feature extraction, deep feature extraction, and high-quality image reconstruction. The model’s long skip connections allow it to send low-frequency data directly to the high-quality image reconstruction module. The process took into account transfer learning. Because only 800 images were used for training, this was done to prevent the model from overfitting.
