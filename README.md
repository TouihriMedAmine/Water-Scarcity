# Water-Scarcity
Enhanced ConvLSTM-Based Drought Forecasting
This project implements a deep learning pipeline using ConvLSTM networks to forecast drought conditions from NASA NLDAS satellite image variables. The model predicts potential drought zones by analyzing time-series map data such as Evapotranspiration, Precipitation, Root Moisture, Surface Soil Moisture, and Vegetation parameters.

📁 Dataset
Source: NASA NLDAS Dataset

Input Format: Grayscale .png images of shape 88x130 pixels per variable

Variables Used:

Evap – Evapotranspiration

Rainf – Rainfall

RootMoist – Root Zone Moisture

SoilM_0_10cm – Surface Soil Moisture (0–10 cm)

TVeg – Transpiration by Vegetation

🏗️ Project Structure
Data Loading: Image sequences loaded per variable, resized and normalized

Preprocessing: Normalization and drought labeling based on low soil moisture percentile

Model Architecture: 2-layer ConvLSTM with Conv2D output layer

Training Strategy: Early stopping, temporal data split, evaluation metrics

GPU Support: MirroredStrategy for multi-GPU acceleration

🚀 Model Architecture
ConvLSTM2D → Dropout → BatchNormalization (x2)

Final Conv2D with sigmoid activation to output drought probability map

Optimizer: Adam, Loss: Binary Crossentropy

🧪 Evaluation
Metrics:

Accuracy

Classification Report

ROC AUC Score

Visualization:

Loss and Accuracy curves

Overlay drought predictions on actual soil moisture maps

🔍 Example Output
Input: 1-day image sequence of 5 variables

Output: 2D map showing predicted drought regions overlaid on soil moisture base map

python
Copy
Edit
drought_overlay, raw_pred, mask = generate_drought_overlay(model, sample_input, base)
⚙️ Requirements
Python ≥ 3.8

TensorFlow ≥ 2.8

OpenCV, NumPy, scikit-learn, Matplotlib

GPU support recommended for training

📦 How to Run
Place the dataset under:
/kaggle/input/water-scarcity/visualization_outputUp/

Train the model:

bash
Copy
Edit
python train_drought_model.py
Visualize predictions:

bash
Copy
Edit
python visualize_output.py
📌 Notes
The drought label is generated if the 3-day forward average of surface soil moisture drops below the 20th percentile.

Adjust SEQUENCE_LENGTH, EPOCHS, and BATCH_SIZE for different forecasting granularities or compute budgets.

Multi-GPU support is enabled via tf.distribute.MirroredStrategy().



For: Drought monitoring and early warning systems using satellite data + deep learning

