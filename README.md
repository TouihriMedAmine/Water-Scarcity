# 💧 Intelligent Climate and Water Forecasting Platform

## 🌍 Project Overview  
We built an integrated intelligent platform for climate forecasting, water level estimation, drought prediction, environmental awareness, and interactive data visualization. The system empowers users to anticipate environmental changes, explore satellite-based predictions, and interact with a chatbot expert in water and climate issues.

---

## 🧠 Core Modules

### 1. Environmental Forecasting by Location & Date  
We developed three deep learning models to predict key environmental variables:  

- 🌡 **Average surface temperature** (`AvgSurfT`)  
- 🌧 **Rainfall** (`Rainf`)  
- 💨 **Potential evapotranspiration** (`PotEvap`)  

Users input geographic coordinates and a future date. The system then generates predictions and passes them to a **Large Language Model (LLM)**, which produces a personalized climate report for the user.

---

### 2. Surface Runoff Prediction via ConvLSTM  
A **ConvLSTM encoder–decoder architecture** forecasts **surface runoff** up to 6 days ahead. It ingests time-series grayscale satellite image data of:  

- **Rainfall**  
- **Soil moisture**  
- **Surface and base flow**  

This module enables advanced hydrological prediction and supports early warning systems for flood risks.

---

### 3. Satellite-Based Water Level Estimation  
Using the **RESISC45** remote sensing dataset, this module classifies satellite images and estimates water levels:  

- 🌊 **Classify water bodies**: lakes, rivers, and harbors  
- 📊 **Predict water levels**: low, medium, or high  

We use a combination of **CNNs**, **U-Net**, **FCN**, and **ResNet** architectures to deliver robust classification and estimation from aerial imagery.

---

### 4. Drought Forecasting with ConvLSTM (Water-Scarcity)  
This deep learning pipeline uses **ConvLSTM networks** to predict drought zones from **NASA NLDAS** satellite variables. It analyzes sequences of grayscale image maps to assess future drought risk based on:  

- 🌿 **Evapotranspiration** (`Evap`)  
- 🌧 **Rainfall** (`Rainf`)  
- 🌱 **Root Zone Moisture** (`RootMoist`)  
- 🌍 **Surface Soil Moisture** (`SoilM_0_10cm`)  
- 🌾 **Vegetation Transpiration** (`TVeg`)  

The model predicts drought probabilities using a **2-layer ConvLSTM** followed by a **Conv2D** output layer. It includes **multi-GPU support**, **ROC-AUC evaluation**, and **overlay visualizations** of drought predictions on real soil moisture maps.

**📁 Dataset**: NASA NLDAS  
**Input**: Grayscale `.png` images (88x130 pixels) per variable  
**Architecture**: `ConvLSTM2D → Dropout → BatchNorm (×2) → Conv2D (sigmoid)`  
**Loss**: Binary Crossentropy, **Optimizer**: Adam  
**Evaluation**: Accuracy, Classification Report, ROC AUC  
**Example**:  
```python
drought_overlay, raw_pred, mask = generate_drought_overlay(model, sample_input, base)
