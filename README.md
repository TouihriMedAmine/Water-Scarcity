## 🌍 Project Overview
We built an integrated intelligent platform for climate forecasting, water level estimation, climate awareness, and interactive data visualization. The system empowers users to anticipate environmental changes, explore satellite-based predictions, and interact with a chatbot expert in water and climate issues.

## 🧠 Core Modules

## 1. Environmental Prediction (based on location and future date)
Three deep learning models predict:

🌡 Average surface temperature (AvgSurfT)

🌧 Rainfall (Rainf)

💨 Potential evapotranspiration (PotEvap)

Using coordinates and a future date as input, predictions are passed to a text generation model (LLM) that outputs a personalized environmental report.

## 2. Surface Runoff Prediction (ConvLSTM)
A ConvLSTM encoder–decoder model forecasts surface runoff 6 days ahead using time-series of hydrometeorological grayscale image data (rainfall, soil moisture, base flow, and surface flow).

## 3. Water Level Estimation from Satellite Images
Using the RESISC45 dataset:

Classify images into lakes, rivers, and harbors

Estimate water levels (low, medium, high) using CNNs, U-Net, FCN, or ResNet

## 4. Droplets — Domain-Specific Chatbot
A climate/water-focused AI chatbot powered by a fine-tuned local LLM. It answers questions about:

Climate change impact

Water use in agriculture, industry, and households

Smart technologies (AI, IoT, desalination)

Sustainability and policy strategies

Its knowledge is based on the ONAGRI 2023 report.

## 5. Unified Web Interface
A user-friendly web page allows:

Input of future date and location

Access to environmental predictions and generated report

Visualization of predictive maps

Interaction with the chatbot

## ⚙️ Tech Stack
Deep Learning: PyTorch, TensorFlow, Keras

Models: ConvLSTM, U-Net, ResNet, LSTM, LLM

Frontend: HTML/CSS/JS, Vue.js or React

Backend: Flask / FastAPI

Chatbot: LangChain, FAISS, Ollama (local LLM)

Data Sources: NASA, RESISC45, ONAGRI Report

