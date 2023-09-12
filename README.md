## Image segmentation with UNet model

Data, predictions and saved model can be found here: https://drive.google.com/drive/folders/1eFxwcWIMcmCI9F7FzI-nLLcA9-BBp-Cq

## Introduction

This project focuses on semantic segmentation of images using the U-Net architecture.

## Prerequisites

* Python: 3.8 or newer
* Libraries: TensorFlow 2.x, NumPy, Matplotlib, Pillow

## Model Architecture

The U-Net architecture is a type of convolutional neural network that is widely used for semantic segmentation tasks. Its architecture is symmetrical and consists of an encoder and a decoder part, which are connected by a bottleneck.

Details of the architecture can be found [here](https://arxiv.org/abs/1505.04597).

## Results

Predictions are made on the validation dataset and the results are saved in the drive link provided above. A small sample of these predictions can be seen in the `results.ipynb` file.

