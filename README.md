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


# 💧 Droplets — Climate & Water Resource Chatbot

**Droplets** is a specialized AI chatbot focused on **climate change**, **water usage**, and **technological innovation in water management**. It is powered by insights extracted from the official [ONAGRI 2023 report](http://www.onagri.nat.tn/uploads/secteur-eau/eau-2023.pdf), processed using natural language tools and embedded for retrieval-based answers.

## 🌍 Core Topics

- Effects of climate change on water resources
- Agricultural, domestic, and industrial water usage in Tunisia
- New technologies for water management (IoT, desalination, AI, sensors, etc.)
- Water policy and sustainability strategies

## 🧠 Chatbot Behavior

- Responds with clear, friendly, and concise answers (max 60 words)
- Stays strictly within the water/climate/technology domain
- Refuses unrelated questions politely and guides the user back to relevant topics
- Uses source documents when answering, and never invents facts
- Indicates uncertainty if information is not found

## 📘 Primary Data Source

- **ONAGRI 2023 Report**: Overview of climate impact, resource usage, and sector evolution  
  [📄 Download PDF](http://www.onagri.nat.tn/uploads/secteur-eau/eau-2023.pdf)

## ⚙️ Tech Stack

- **LangChain** for orchestration
- **FAISS** for semantic document retrieval
- **Ollama** for running the local **`mystral`** model
- **Python 3.10+**
  - `langchain_community`
  - `langchain_ollama`
  - `langdetect`

## 🚀 Setup Instructions

1. **Start Ollama**  
   Make sure `ollama` is installed and running:
   ```bash
   ollama serve






















