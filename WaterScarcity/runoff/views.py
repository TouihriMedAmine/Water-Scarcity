import os
from django.shortcuts import render
from django.conf import settings
from .forms import RunoffPredictionForm
from .utils import infer_and_plot_runoff_plain
from uuid import uuid4
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def predict_runoff(request):
    if request.method == 'POST':
        try:
            # Example: Get uploaded images
            images = [request.FILES.get(f'image_day{i}') for i in range(6)]

            # Check if all images are present
            if not all(images):
                return JsonResponse({'error': 'Please upload all 6 images.'})

            # Save the uploaded images temporarily
            image_paths = []
            for idx, image in enumerate(images):
                image_name = f"temp_image_day{idx}.png"
                image_path = os.path.join(settings.MEDIA_ROOT, image_name)
                with open(image_path, 'wb') as f:
                    for chunk in image.chunks():
                        f.write(chunk)
                image_paths.append(image_path)

            # Call the inference function
            W0, H0 = 128, 128  # Define your image size for the model
            extent = (-180, 180, -90, 90)  # Example extent (you may need to adjust this)
            weights_path = os.path.join(settings.BASE_DIR, 'model_weights.pt')  # Path to your model weights

            pred_norm, result_image_path = infer_and_plot_runoff_plain(
                image_paths=image_paths,
                weights_path=weights_path,
                W0=W0,
                H0=H0,
                extent=extent
            )

            # Remove temporary images after processing
            for image_path in image_paths:
                os.remove(image_path)

            # Generate runoff stats
            runoff_stats = {
                'runoff_min': float(pred_norm.min()),
                'runoff_max': float(pred_norm.max()),
                'runoff_mean': float(pred_norm.mean()),
            }

            # Return the result image URL and stats
            result_image_url = os.path.join(settings.MEDIA_URL, os.path.basename(result_image_path))
            return JsonResponse({
                'image_url': result_image_url,
                **runoff_stats
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
