
import os
import sys
from uuid import uuid4
from django.conf import settings
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def infer_and_plot_runoff_plain(
    image_paths: list,
    weights_path: str,
    W0: int,
    H0: int,
    extent: tuple,
    T: int = 6,
    hid_ch: int = 8,
    k: int = 3
) -> np.ndarray:
    """
    Similar to the original, but without displaying the colorbar (scale).
    - image_paths  : List of T PNG images.
    - weights_path : Path to model weights (best_qs_only.pth or model_final_qs_only.pth).
    - W0, H0       : Input dimensions for the model.
    - extent       : (xmin, xmax, ymin, ymax) for imshow.
    - T, hid_ch, k : Hyperparameters.
    Returns: pred_norm (shape H0 x W0) and image path.
    """
    weights_path = os.path.join(settings.BASE_DIR, 'runoff', 'model', 'best_qs_only.pth')
    # 1) Import model
    code_dir = os.path.dirname(os.path.abspath(weights_path))
    sys.path.insert(0, code_dir)
    from .models import ConvLSTMForecaster

    # 2) Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvLSTMForecaster(in_ch=1, hid_ch=hid_ch, k=k, T=T).to(device).eval()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    # 3) Prepare images
    X_list = []
    for p in image_paths:
        im = Image.open(p).convert('L').resize((W0, H0), Image.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.0
        X_list.append(arr)
    X = np.stack(X_list).reshape(T, 1, H0, W0)
    Xb = torch.tensor(X).unsqueeze(0).to(device)

    # 4) Inference
    with torch.no_grad():
        pred_norm = model(Xb).cpu().squeeze().numpy()

    # 5) Save output image
    data = np.flipud(pred_norm)
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10,6), facecolor='white')
    cmap = LinearSegmentedColormap.from_list('dark_blue',['#000033','#ffffff'])
    ax.imshow(
        data,
        origin='lower',
        extent=extent,
        vmin=0, vmax=1,
        cmap=cmap,
        aspect='auto'
    )
    ax.set(
        xlabel='Longitude',
        ylabel='Latitude',
        title=f'Predicted Surface Runoff at T+{T} days (normalized)'
    )
    plt.tight_layout()
    
    # Save result
    result_image_path = os.path.join(settings.MEDIA_ROOT, f'runoff_result_{uuid4()}.png')
    plt.savefig(result_image_path)
    plt.close()

    return pred_norm, result_image_path
