# Water Level Estimation 

## Overview

This project focuses on estimating water levels from remote sensing images using deep learning techniques. Specifically, we classify images of lakes, rivers, and harbors, and then predict water levels for each category based on the image features.

The project utilizes the RESISC45 dataset, which is a publicly available benchmark dataset for remote sensing image scene classification. It contains 31,500 images across 45 scene classes, with 700 images per class. For this project, we focus on three specific scene classes: lakes, rivers, and harbors.


## Models Developed

### 1.Image Classification:

Classifies images into three categories: lakes, rivers, and harbors.

### 2.Water Level Estimation for Lakes:

Uses CNN (Convolutional Neural Networks) to estimate the water level based on image features.

### 3.Water Level Estimation for Rivers:

Explores U-Net, FCN (Fully Convolutional Network), and ResNet architectures to predict water levels based on river images.

### 4.Water Level Estimation for Harbors:

Similar to rivers, uses U-Net, FCN, and ResNet to estimate water levels in harbor images.


## Dataset

The RESISC45 dataset, created by Northwestern Polytechnical University (NWPU), is used for this project. It contains images in 45 scene categories, and we are specifically working with the lakes, rivers, and harbors classes, which provide 700 images per category.


## Workflow

### 1.Image Classification: 
Images are first classified into one of the three categories: lake, river, or harbor.

### 2.Water Level Estimation: 
Once classified, water levels (low, medium, high) are predicted for each category using the respective model architectures.


## Technologies Used

Deep Learning: TensorFlow/Keras for model building and training.

CNN for water level estimation in lakes.

U-Net, FCN, and ResNet for water level estimation in rivers and harbors.

RESISC45 dataset for image data.























