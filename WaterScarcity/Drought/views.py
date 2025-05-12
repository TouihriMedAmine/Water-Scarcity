from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .forms import DroughtPredictionForm
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from django.core.files.storage import default_storage
from tensorflow.keras.models import load_model
from .utils import preprocess_single_images  # Ensure this handles file paths

# Load model and setup
MODEL_PATH = os.path.join(settings.BASE_DIR, 'Drought', 'model', 'drought-prediction.h5')
BASEMAP_PATH = os.path.join(settings.BASE_DIR,'Drought', 'drought_predictions.png')
RESULT_FOLDER = os.path.join(settings.MEDIA_ROOT, 'drought_results')
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)

from django.http import JsonResponse

@csrf_exempt
def predict_drought(request):
    if request.method == 'POST':
        try:
            # Retrieve each file
            evap_file = request.FILES.get('Evap')
            rainf_file = request.FILES.get('Rainf')
            rootmoist_file = request.FILES.get('RootMoist')
            soilm_file = request.FILES.get('SoilM_0_10cm')
            tveg_file = request.FILES.get('TVeg')

            print(request.FILES)  # Debug

            # Check for missing files
            if not all([evap_file, rainf_file, rootmoist_file, soilm_file, tveg_file]):
                return JsonResponse({'error_message': 'Missing one or more required files for prediction.'})

            # Preprocess all 5 images (they should be combined inside this function)
            input_tensor = preprocess_single_images({
                    'Evap': evap_file,
                    'Rainf': rainf_file,
                    'RootMoist': rootmoist_file,
                    'SoilM_0_10cm': soilm_file,
                    'TVeg': tveg_file
                })
  # Expected shape: (1, 88, 130, 5) or similar

            input_tensor = np.expand_dims(input_tensor, axis=1)  # Adjust shape if needed

            # Predict
            prediction = model.predict(input_tensor)
            pred_map = prediction[0, :, :, 0]  # Adjust indexing if needed

            # Load base map and overlay prediction
            base_map = mpimg.imread(BASEMAP_PATH)

            plt.figure(figsize=(10, 6))
            plt.imshow(base_map, extent=[0, pred_map.shape[1], pred_map.shape[0], 0])
            plt.imshow(pred_map, cmap='hot', alpha=0.6, extent=[0, pred_map.shape[1], pred_map.shape[0], 0])
            plt.colorbar(label='Drought Probability')
            plt.title("Predicted Drought Map")

            # Save result
            result_path = os.path.join(RESULT_FOLDER, 'result.png')
            plt.savefig(result_path)
            plt.close()

            return JsonResponse({
                'image': settings.MEDIA_URL + f'drought_results/result.png',
            })

        except Exception as e:
            return JsonResponse({'error_message': str(e)})

    return JsonResponse({'error_message': 'Invalid form submission.'})
