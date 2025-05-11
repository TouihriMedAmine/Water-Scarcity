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

### 5. Precipitation Prediction and Classification
This module forecasts precipitation intensity and visualizes it through class segmentation and quantitative analysis. It includes:

**Temporal Prediction:**
A **PredRNN++ (PredRNNv2)** model trained on historical precipitation heatmaps to generate future precipitation sequences.

Spatial Segmentation:
A **U-Net** model segments each precipitation heatmap into four classes:

🚫**No Rain**

🌦 **Light Rain**

🌧 **Moderate Rain**

🌩 **Heavy Rain**

**Statistical Output:**
The system calculates precipitation volume, class-wise area coverage, and temporal trends to assist in hydrological planning and disaster risk assessment.

This module enhances rainfall-specific modeling by offering both visual and numerical insights into upcoming precipitation events.

### 6. Deforestation Forecasting with U-Net (Forest‐Cover Change)

This deep learning pipeline employs a 2D **U-Net** to forecast one-year-ahead **NDVI** and flag likely deforestation from current Sentinel-2 imagery. It processes 256×256px multispectral patches (normalized reflectance) and outputs high-resolution change masks:

🌲 **Sentinel-2 Bands:** Red (B4), Green (B3), Blue (B2), NIR (B8)

🌱 **NDVI Regression:** model learns to predict next-year NDVI from four-band inputs

🔗 **U-Net Architecture:** three down-sample

🛠 Post-Processing: compute **ΔNDVI = NDVIₚᵣₑₐd − NDVIₙ**, threshold (Δ < −0.1) to generate a binary deforestation mask

📊 **Evaluation**: MSE on held-out AOIs, visual overlay of predicted loss on true imagery

This module integrates seamlessly with the overall forecasting system, providing actionable maps of emerging forest loss ready for GIS export and alerting.

---

### 7. Droplets — Domain-Specific Chatbot
A climate/water-focused AI chatbot powered by a fine-tuned local LLM. It answers questions about:

Climate change impact

Water use in agriculture, industry, and households

Smart technologies (AI, IoT, desalination)

Sustainability and policy strategies

Its knowledge is based on the ONAGRI 2023 report.

---

### 8. Unified Web Interface

A user-friendly web page allows:

Input of future date and location

Access to environmental predictions and generated report

Visualization of predictive maps

Interaction with the chatbot

---

## ⚙️ Tech Stack
Deep Learning: PyTorch, TensorFlow, Keras

Models: ConvLSTM, U-Net, ResNet, LSTM, LLM,PredRNNv2

Frontend: HTML/CSS/JS, Vue.js or React

Backend: Flask / FastAPI

Chatbot: LangChain, FAISS, Ollama (local LLM)

Data Sources: NASA, RESISC45, ONAGRI Report

