**📍 Predictive System Based on Location and Future Date** 
 **Models to Train** 
 **1. Environmental Attribute Predictive Models (3 models)** 
 **Objective:** Predict the future value of a specific environmental attribute for a given location (latitude, longitude) and date. 
 **Target Attributes:** 
 Average Surface Temperature (AvgSurfT) 
 Rainfall (Rainf) 
 Potential Evapotranspiration (PotEvap) 
 **Approach:** 
 Each attribute will be predicted by a dedicated deep learning model. 
 Envisioned architectures: RNN, LSTM, or U-Net. 
 Model inputs: latitude, longitude, date. 
 Models will be trained on historical time series data, associating (latitude, longitude, date) with attribute values. 
 
 
 **2. Text Generation Model (LLM)** 
 **Objective:** Produce a structured textual report for the user. 
 **Approach:** 
 The model will take as input: 
 The AvgSurfT prediction 
 The Rainf prediction 
 The PotEvap prediction 
 The coordinates and date 
 The LLM will be a pre-trained language model, fine-tuned specifically for this task. 
 It will generate a report explaining: 
 The predicted future environmental conditions 
 Their potential impact 
 Possible recommendations for actions or precautions 
 
 
 **System Usage Phase** 
 **User Data Input:** 
 The user provides only the location (latitude, longitude) and a future date. 
 **Environmental Attribute Prediction:** 
 Each model respectively predicts AvgSurfT, Rainf, and PotEvap for the given location and date. 
 **Report Generation:** 
 The LLM uses the three predictions to generate a personalized text report including: 
 Predicted average temperature 
 Estimated rainfall 
 Anticipated irrigation needs 
 Contextualized agricultural advice 
 
 
 **🔥 Final Pipeline Summary** 
 **Inputs:** Latitude, Longitude, Date 
 **Models:** 
 Model 1: AvgSurfT Prediction 
 Model 2: Rainf Prediction 
 Model 3: PotEvap Prediction 
 Model 4: Report Generation LLM 
 **Output:** Personalized textual report for the user