# 🌍 Water Scarcity Forecasting Platform

## Overview
An integrated intelligent platform for climate forecasting, water level estimation, drought prediction, environmental awareness, and interactive data visualization. Built with Python, TensorFlow, Keras, PyTorch, React, and Flask, this project empowers users to anticipate environmental changes, explore satellite-based predictions, and interact with a domain-specific chatbot.

## Features
- **Environmental Forecasting by Location & Date**
  - Predicts average surface temperature, rainfall, and potential evapotranspiration for any location and date using deep learning models (U-Net, LSTM).
  - Integrates predictions with a Large Language Model (LLM) to generate personalized climate reports.
  - Supports input of geographic coordinates and future dates for tailored forecasts.

- **Surface Runoff Prediction**
  - Utilizes a ConvLSTM encoder–decoder to forecast surface runoff up to 6 days ahead.
  - Processes time-series satellite images of rainfall, soil moisture, and flow variables.
  - Provides early warning for flood risks and hydrological planning.

- **Satellite-Based Water Level Estimation**
  - Classifies water bodies (lakes, rivers, harbors) from satellite imagery using CNN, U-Net, FCN, and ResNet architectures.
  - Estimates water levels (low, medium, high) for each detected water body.
  - Enables robust monitoring and management of water resources.

- **Drought Forecasting**
  - Predicts drought zones using NASA NLDAS satellite variables (evapotranspiration, rainfall, root zone moisture, soil moisture, vegetation transpiration).
  - Employs a 2-layer ConvLSTM and Conv2D output for spatial drought probability maps.
  - Includes multi-GPU support, ROC-AUC evaluation, and overlay visualizations.

- **Precipitation Prediction and Classification**
  - Forecasts precipitation intensity and generates future precipitation sequences with PredRNNv2.
  - Segments precipitation heatmaps into four classes (no rain, light, moderate, heavy) using U-Net.
  - Computes precipitation volume, area coverage by class, and temporal trends for risk assessment.

- **Deforestation Forecasting with U-Net**
  - Predicts next-year NDVI and flags likely deforestation from Sentinel-2 multispectral imagery.
  - Outputs high-resolution change masks and binary deforestation alerts (ΔNDVI < -0.1).
  - Supports GIS export and visual overlays for actionable forest loss monitoring.

- **Droplets — Domain-Specific Chatbot**
  - AI chatbot specialized in climate and water issues, powered by a fine-tuned local LLM.
  - Answers questions on climate change, water use, smart technologies, and sustainability.
  - Knowledge base includes ONAGRI 2023 report and domain-specific resources.

- **Unified Web Interface**
  - User-friendly web page for entering location and date, accessing predictions, and visualizing results.
  - Interactive chatbot and map overlays for enhanced user experience.
  - Supports data export and integration with other tools.

## Tech Stack
### Frontend
- HTML/CSS/JS

### Backend
- Python
- Django
- TensorFlow, Keras, PyTorch
- LangChain, FAISS, Ollama (LLM),Mistral (LLM)
- U-Net, PredRNNv2, ConvLSTM, Resnet, CNN,


### Other Tools
- NASA NLDAS, RESISC45, ONAGRI datasets
- Git for version control
- Sentinel-2 satellite imagery

## Directory Structure
- `WaterScarcity/` – Django app modules (Drought, Irrig, Waterlevel, Watershed, runoff, chat, etc.)
- `Models_KERAS/`, `Model_Water_Level/` – Model files and integration scripts
- `Precepitation_needs/` – Precipitation data
- `static/`, `templates/` – Web assets
- `uploads/`, `results/` – Data and output storage

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Water-Scarcity.git


## Acknowledgments
- **NASA**: For providing open-access satellite and climate data (NLDAS, remote sensing imagery) essential for model training and evaluation.
- **ONAGRI (Observatoire National de l'Agriculture, Tunisia)**: For their comprehensive climate and water management reports, which informed the chatbot's knowledge base and project context.
- **RESISC45 Dataset**: For high-quality remote sensing images used in water body classification and water level estimation modules.
- **Open Source Community**: For the development and maintenance of key libraries and frameworks, including TensorFlow, PyTorch, Keras, Flask, FastAPI, React, Vue.js, and LangChain.
- **Academic Publications**: For research on ConvLSTM, U-Net, PredRNNv2, and other deep learning architectures that inspired the project's models.