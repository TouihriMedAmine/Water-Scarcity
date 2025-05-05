# 🌍 Project Overview  
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

---

### 5. Droplets — Domain-Specific Chatbot
A climate/water-focused AI chatbot powered by a fine-tuned local LLM. It answers questions about:

Climate change impact

Water use in agriculture, industry, and households

Smart technologies (AI, IoT, desalination)

Sustainability and policy strategies

Its knowledge is based on the ONAGRI 2023 report.

---

### 6. Unified Web Interface

A user-friendly web page allows:

Input of future date and location

Access to environmental predictions and generated report

Visualization of predictive maps

Interaction with the chatbot

---

## ⚙️ Tech Stack
Deep Learning: PyTorch, TensorFlow, Keras

Models: ConvLSTM, U-Net, ResNet, LSTM, LLM

Frontend: HTML/CSS/JS, Vue.js or React

Backend: Flask / FastAPI

Chatbot: LangChain, FAISS, Ollama (local LLM)

Data Sources: NASA, RESISC45, ONAGRI Report

