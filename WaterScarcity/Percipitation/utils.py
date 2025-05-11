import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from rasterio import features
import rasterio
import torch
from .save_images import save_images,visualize_predictions_as_gif
# Assuming these are defined somewhere
from .PredRNN import PredRNN  # your model class
from .Unet import UNet  # your model class
import os
from torchvision import transforms
from PIL import Image
# Load once and reuse
GDF = gpd.read_file("static/us-state-boundaries/us-state-boundaries.shp")  # Update this path as needed

def rasterize_state_mask(state_geom, out_shape=(256,256), bounds=(-125.0, 24.0, -66.0, 50.0) ):
    if bounds is None:
        bounds = state_geom.bounds  # Use actual shape bounds if not provided

    transform = rasterio.transform.from_bounds(*bounds, out_shape[1], out_shape[0])
    mask = features.rasterize(
        [(state_geom, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )
    return mask.astype(bool), transform

def get_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_dates = []

    with torch.no_grad():
        for images, _, dates in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.squeeze(1).cpu().numpy()  # [B, H, W]

            all_preds.extend(preds)
            all_dates.extend(dates)

    return np.array(all_preds), all_dates

def extract_trends_by_state(predicted_masks, state_name, precipitation_bounds=(-125.0, 24.0, -66.0, 50.0), 
                          save_debug_overlays=True, save_dir="static/debug_overlays", max_debug_plots=5):
    import os
    os.makedirs(save_dir, exist_ok=True)

    class_intensities = {0: 0, 1: 5, 2: 15, 3: 30}   # mm per class (as array for vectorization)
    class_weights=[4.5, 3.25, 2, 0]
    # Debug: Verify input shapes
    print(f"Input shapes - Predicted: {predicted_masks.shape}, " 
          f"Bounds: {precipitation_bounds}")

    # Get state geometry
    pred_classes = torch.argmax(predicted_masks, dim=1)  # [T, H, W]
    
    # Get state mask
    state_geom = GDF[GDF['name'].str.lower() == state_name.lower()].geometry.values[0]
    H, W = predicted_masks.shape[2:]
    transform = rasterio.transform.from_bounds(*precipitation_bounds, W, H)
    state_mask = features.rasterize(
        [(state_geom, 1)],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    ).astype(bool)
    
    # Vectorized calculation
    class_counts = torch.zeros((pred_classes.shape[0], 4), device=pred_classes.device)
    for class_id in range(4):
        class_counts[:, class_id] = torch.sum(
            (pred_classes == class_id) & state_mask,
            dim=(1, 2)
        )
        plot_class_distribution(pred_classes, state_mask, t=class_id, save_path=os.path.join(save_dir, f"{state_name}_statemask{class_id}.png"))
    
    # Convert to numpy and calculate precipitation
    class_counts_np = class_counts.cpu().numpy()
    precipitation = np.dot(class_counts_np, class_weights) / 100  # Your normalization
    plot_results(precipitation, class_counts_np, dates=None, save_path=os.path.join(save_dir, f"{state_name}_class_distribution.png"))
    
    return precipitation

def plot_results(precipitation, class_counts, dates=None,save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Precipitation plot
    ax1.plot(precipitation)
    ax1.set_title('Precipitation Trend')
    ax1.set_ylabel('Normalized Units')
    
    # Class distribution plot
    for class_id in range(4):
        ax2.plot(class_counts[:, class_id], label=f'Class {class_id}')
    ax2.set_title('Class Distribution Over Time')
    ax2.set_ylabel('Pixel Count')
    ax2.legend()
    
    if dates:
        ax1.set_xticks(range(len(dates)))
        ax1.set_xticklabels(dates, rotation=45)
        ax2.set_xticks(range(len(dates)))
        ax2.set_xticklabels(dates, rotation=45)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close
def plot_class_distribution(pred_classes, state_mask, t=0,save_path=None):
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.imshow(pred_classes[t], vmin=0, vmax=3, cmap='jet')
    plt.title("All Class Predictions")
    
    plt.subplot(122)
    masked = np.ma.masked_where(~state_mask, pred_classes[t])
    plt.imshow(masked, vmin=0, vmax=3, cmap='jet')
    plt.title("State-Masked Classes")
    plt.colorbar(label='Class ID')
    
    plt.savefig(save_path)
    plt.close()

def plot_state_trend(state_name, trend_data, dates, save_path=None):
    if len(dates) < len(trend_data):
        # Create a date range with len(trend_data) periods, hourly intervals
        dates = pd.date_range(start='2000-01-01', periods=len(trend_data), freq='h')
    else:
        dates = pd.to_datetime(dates, format='%Y%m%d%H')
    print(len(trend_data), len(dates))
    df = pd.DataFrame({
        'date': dates,
        'total_precipitation': trend_data
    })
    print(len(trend_data), len(dates))
    print(dates)
    df.set_index('date', inplace=True)
    weekly_trend = df

    plt.figure(figsize=(10, 4))
    plt.plot(weekly_trend.index, weekly_trend['total_precipitation'], label=f"{state_name} - Total Precipitation")
    plt.xlabel("Date")
    plt.ylabel("Total Precipitation (kg/m2)")
    plt.title(f"Precipitation Trend in {state_name} (mm/hr)")
    plt.legend()
    plt.grid(True)

    if not save_path.startswith("static/"):
        save_path = "static/" + save_path
    if save_path:
        plt.savefig(save_path)
        
    else:
        plt.show()



#PREDRNNv2

class Config:
    img_width = 128  # Changed from 128 to match your input size
    patch_size = 1
    img_channel = 3
    filter_size = 5
    stride = 1
    layer_norm = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_length = 6
    input_length = 3
    reverse_scheduled_sampling = 0
    decouple_beta = 0.1
    visual = 0
    visual_path = "./visual"
    lr = 1e-4
    epochs = 10

def run_inference(model_path, input_sequence, num_samples=1):
    configs = Config()
    device = torch.device(configs.device)

    # Load model
    model = PredRNN(num_layers=4, num_hidden=[64, 64, 64, 64], configs=configs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    

    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        all_pred_frames = []
        all_gt_frames = []
        for idx in range(input_sequence.shape[0]):
            if idx >= num_samples:
                break
            
            print(f"Input sequence shape: {input_sequence.shape}")
            # 1. Prepare input with dummy frame
            real_frames = input_sequence[idx][:3].unsqueeze(0).to(device)  # [1, 3, H, W, C]
            dummy_frames = torch.zeros_like(real_frames[:, :3])  # 3 dummy frames
            batch = torch.cat([real_frames, dummy_frames], dim=1)   # Shape: [1, 4, H, W, C]
            print(f"Batch shape: {batch.shape}")
            # 2. Set mask to force autoregressive prediction
            mask_true = torch.zeros(1, 3, 1, 1, 1).to(device)  # All zeros = use predictions
            print(f"Mask shape: {mask_true.shape}")
            # 3. Run model
            output, _ = model(batch, mask_true)  # Output shape: [1, , H, W, C]
            pred_frames = output[:, 2:]  
            print(f"Output shape: {output.shape}")
            full_sequence = torch.cat([
                real_frames,  # Input frames 0-2
                pred_frames   # Predicted frames 3-5
            ], dim=1)  # [1, 6, H, W, C]
            
            # 6. Store results
            all_pred_frames.append(full_sequence)
            all_gt_frames.append(batch[:,1:])  
        
                
    # Prepare for visualization
    if all_pred_frames:
        # Stack along batch dimension
        combined_preds = torch.cat(all_pred_frames, dim=1)  # [B, 3+N, H, W, C]
        combined_gts = torch.cat(all_gt_frames, dim=1)      # [B, 3+N, H, W, C]
        
        visualize_predictions_as_gif(
            combined_preds, 
            combined_gts, 
            "static/img/empty_map_us.png",
            save_dir="static/model_gifs"
        )
        
        return combined_preds, combined_gts
    
def run_unet(preds,state,image_dates):
    

    batch = preds[0]  # Shape: [8, 3, 256, 256]
    print(f"Batch shape: {batch.shape}")
    model = UNet(n_channels=3, n_classes=4)  # Adjust class count if needed
    model.load_state_dict(torch.load('static/modelsP/unet.pth', map_location='cpu'))
    model.eval()

    with torch.no_grad():
        predictions = model(batch)  # [8, C, 256, 256]

    predicted_masks = predictions  # [8, 256, 256]

    # Analyze and plot trends
    trend = extract_trends_by_state(predicted_masks, state)
    plot_path = f'trends/{state}.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plot_state_trend(state, trend, image_dates, save_path=plot_path)
    
    return trend, image_dates,plot_path