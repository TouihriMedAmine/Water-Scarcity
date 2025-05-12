import os
import numpy as np
import torch
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import UNet2D  # Adjust this if your model is in another location
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet2D(in_channels=4, out_channels=1)
model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, 'Watershed', 'models', 'unet_channels.pth'), map_location=device))
model.to(device)
model.eval()

def predict_mask_and_overlay(input_path, output_path, thresh=0.68, n_channels=4):
    patch = np.load(input_path).astype(np.float32)

    # Ensure the patch is in CHW format
    if patch.ndim == 3 and patch.shape[-1] == n_channels:  # HWC → CHW
        p = patch.transpose(2, 0, 1)
    else:
        p = patch

    tensor = torch.from_numpy(p)[None, ...].to(device)

    with torch.no_grad():
        out = model(tensor).squeeze().cpu().numpy()
    mask = (out > thresh).astype(np.uint8)

    # Extract NDVI (last channel)
    if patch.ndim == 3 and patch.shape[-1] == 4:
        ndvi = patch[..., -1]  # HWC
    else:
        ndvi = patch[-1, ...]  # CHW

    # Keep right half
    h, w = mask.shape
    x_mid = w // 2
    ndvi_right = ndvi[:, x_mid:]
    mask_right = mask[:, x_mid:]

    # Plot overlay
    cmap_mask = ListedColormap(['none', 'red'])
    fig, ax = plt.subplots(figsize=(6, 6))
    im0 = ax.imshow(ndvi_right, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.imshow(mask_right, cmap=cmap_mask, alpha=0.6, vmin=0, vmax=1, interpolation='nearest')

    ax.set_title("Masque de déforestation", fontsize=14)
    ax.axis('off')
    cbar = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI", rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Collect statistics
    stats = {
        'ndvi_min': float(np.min(ndvi_right)),
        'ndvi_max': float(np.max(ndvi_right)),
        'ndvi_mean': float(np.mean(ndvi_right)),
        'deforestation_pixels': int(np.sum(mask_right)),
    }
    return stats

@require_POST
def predict_watershed(request):
    try:
        # Get the uploaded file
        uploaded_file = request.FILES['photo_watershed']
        
        # Check file format
        if not uploaded_file.name.endswith('.npy'):
            return JsonResponse({'error': 'Invalid file format. Please upload a .npy file.'}, status=400)

        # Save the uploaded file
        npy_path = default_storage.save(f'uploads/{uploaded_file.name}', uploaded_file)
        full_npy_path = os.path.join(settings.MEDIA_ROOT, npy_path)

        # Prepare output file paths
        output_filename = f"ndvi_overlay_{uploaded_file.name.replace('.npy', '.png')}"
        output_path = os.path.join(settings.MEDIA_ROOT, 'results', output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Run model prediction and overlay generation
        stats = predict_mask_and_overlay(full_npy_path, output_path)

        # Build the URL to the result image
        image_url = settings.MEDIA_URL + f'results/{output_filename}'

        # Return the result as JSON response
        return JsonResponse({**stats, 'image_url': image_url})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
