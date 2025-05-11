import torch
import torch.nn as nn
from .model import ConvLSTM

import os
import imageio
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import torchvision.utils as vutils

def adjust_pixel_intensity(image, adjustment_value):
    """
    Manually adjust the intensity of each pixel by adding a fixed value.
    
    Parameters:
        image (Tensor): The predicted image tensor with shape (batch_size, channels, height, width).
        adjustment_value (float): The value to add to each pixel's intensity.
        
    Returns:
        Tensor: The adjusted image tensor.
    """
    # Add the adjustment value to each pixel in the image
    adjusted_image = image + adjustment_value
    
    # Optionally, clip the image values to be within a valid range (e.g., [0, 1] for normalized RGB)
    #adjusted_image = torch.clamp(adjusted_image, min=0.0, max=1.0)
    
    return adjusted_image






def save_images(predicted_frames, target_frames, batch_idx):
    """
    Generate side-by-side GIFs comparing predicted and target frames with text labels and custom background.
    """

    output_dir = "./static/model_gifs"
    os.makedirs(output_dir, exist_ok=True)

    # Denormalize from [-1, 1] to [0, 1]
    predicted_frames = (predicted_frames * 0.5 + 0.5).clamp(0, 1)
    target_frames = (target_frames * 0.5 + 0.5).clamp(0, 1)

    # Save entire batch as images (optional)
    
    print(f"predicted_frames shape: {predicted_frames.shape}")

    batch_size, channels, height, width = predicted_frames.shape

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    def make_white_transparent(image):
        """Make near-white pixels fully transparent."""
        image = image.convert("RGBA")
        datas = image.getdata()
        newData = []
        for item in datas:
            if item[0] > 220 and item[1] > 220 and item[2] > 220:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        image.putdata(newData)
        return image

    def add_label_with_background(image, label, background_path, size, label_height=20):
        width, height = size
        # Create full transparent canvas
        labeled = Image.new("RGBA", (width, height + label_height), (255, 255, 255, 0))

        # Load and resize background to match only the image (not the label area)
        background = Image.open(background_path).convert("RGBA").resize((width, height))

        # Place background only behind the image region
        labeled.paste(background, (0, label_height))

        # Prepare image
        image = make_white_transparent(image.convert("RGBA"))
        labeled.paste(image, (0, label_height), image)

        # Draw label on transparent top area
        draw = ImageDraw.Draw(labeled)
        draw.text((width // 2 - 30, 2), label, fill=(0, 0, 0, 255), font=font)

        return labeled

    comparison_frames = []

    for i in range(batch_size):
        pred_img = to_pil_image(predicted_frames[i].cpu())
        tgt_img = to_pil_image(target_frames[i].cpu())

        tgt_labeled = add_label_with_background(tgt_img, "Target", "empty_map_us.png", (width, height))
        pred_labeled = add_label_with_background(pred_img, "Prediction", "empty_map_us.png", (width, height))

        combined = Image.new("RGB", (2 * width, height + 20))
        combined.paste(tgt_labeled.convert("RGB"), (0, 0))
        combined.paste(pred_labeled.convert("RGB"), (width, 0))

        comparison_frames.append(combined)

    gif_path = os.path.join(output_dir, f"comparison_batch{batch_idx}.gif")
    imageio.mimsave(gif_path, comparison_frames, duration=200, loop=0)
    print(f"✅ Saved labeled comparison GIF for sample: {gif_path}")
    
    
    
    
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageEnhance  # For GIF creation
import imageio
from PIL import ImageFilter
from PIL import ImageEnhance

def enhance_brightness(image, factor=1.2):
    return ImageEnhance.Brightness(image).enhance(factor)

def enhance_contrast(image, factor=1.3):
    return ImageEnhance.Contrast(image).enhance(factor)

def enhance_sharpness(image, factor=2.0):
    return ImageEnhance.Sharpness(image).enhance(factor)

def apply_unsharp_mask(image):
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

def improve_image_quality(image):
    image = enhance_brightness(image, factor=1.3)
    image = enhance_contrast(image, factor=1.4)
    image = enhance_sharpness(image, factor=2.0)
    image = apply_unsharp_mask(image)
    return image

def adjust_pixel_intensity(image, factor):
    """
    Adjust the brightness of a PIL image or NumPy array by a given factor.
    factor < 1.0 darkens the image, factor > 1.0 brightens it.
    """
    if isinstance(image, Image.Image):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    elif isinstance(image, np.ndarray):
        adjusted = image * factor
        return np.clip(adjusted, 0, 1)
    else:
        raise ValueError("Unsupported image type for brightness adjustment.")
def make_white_transparent(image):
        """Make near-white pixels fully transparent."""
        image = image.convert("RGBA")
        datas = image.getdata()
        newData = []
        for item in datas:
            if item[0] > 220 and item[1] > 220 and item[2] > 220:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        image.putdata(newData)
        return image


def create_gif(images, file_path, duration=200):
    """
    Create a GIF from a sequence of images.
    
    Args:
        images (list): List of PIL Image objects
        file_path (str): Output file path
        duration (int): Duration between frames in milliseconds
    """
    # Save using imageio
    imageio.mimsave(file_path, images, duration=duration,loop=0)
    print(f"Saved GIF: {file_path}")

def visualize_predictions_as_gif(predictions, ground_truth, background_image_path=None, save_dir="prediction_gifs"):
    """
    Save predicted vs ground truth frames as GIFs overlaid on background image.
    
    Args:
        predictions (Tensor): Shape [B, T, C, H, W]
        ground_truth (Tensor): Shape [B, T, C, H, W]
        background_image_path (str): Path to background image (optional)
        save_dir (str): Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load background image if provided
    background = None
    if background_image_path and os.path.exists(background_image_path):
        background = Image.open(background_image_path).convert('RGB')
        bg_width, bg_height = 512, 256
        background = background.resize((bg_width, bg_height), Image.BICUBIC)


    predictions = predictions.cpu().detach().numpy()
    ground_truth = ground_truth.cpu().detach().numpy()
    print(f"predictions shape2: {predictions.shape}, ground_truth shape2: {ground_truth.shape}")
    
    for i in range(min(predictions.shape[0], 5)):
        # Process predicted frames
        pred_frames = predictions[i]
        pred_images = []
        for t in range(pred_frames.shape[0]):
            # Convert prediction frame to PIL Image
            frame = np.transpose(pred_frames[t], (1, 2, 0))
            
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)
            frame_img = Image.fromarray(frame)
            frame_img = make_white_transparent(frame_img.convert("RGBA"))
            if(t>2):
                frame_img = improve_image_quality(frame_img)
            frame_img = frame_img.resize((bg_width, bg_height), Image.NEAREST)

            if background is not None:
                # Create a copy of the background
                bg_copy = background.copy()
                # Paste prediction frame over background
                bg_copy.paste(frame_img, (0, 0), frame_img.convert('RGBA') if frame_img.mode != 'RGBA' else frame_img)
                pred_images.append(bg_copy)
            else:
                pred_images.append(frame_img)
        
        # Process ground truth frames
        gt_frames = ground_truth[i]
        gt_images = []
        for t in range(gt_frames.shape[0]):
            # Convert ground truth frame to PIL Image
            frame = np.transpose(gt_frames[t], (1, 2, 0))
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)
            frame_img = Image.fromarray(frame)
            frame_img = frame_img.resize((bg_width, bg_height), Image.NEAREST)

            
            if background is not None:
                # Create a copy of the background
                bg_copy = background.copy()
                # Paste ground truth frame over background
                bg_copy.paste(frame_img, (0, 0), frame_img.convert('RGBA') if frame_img.mode != 'RGBA' else frame_img)
                gt_images.append(bg_copy)
            else:
                gt_images.append(frame_img)
        
        # Save as separate GIFs
        pred_path = os.path.join(save_dir, f"sample_{i+1}_predicted.gif")
        gt_path = os.path.join(save_dir, f"sample_{i+1}_groundtruth.gif")
        
        create_gif(pred_images, pred_path)
        create_gif(gt_images, gt_path)

        # Create side-by-side comparison GIF
        if background is not None:
            combined_images = []
            for pred_img, gt_img in zip(pred_images, gt_images):
                # Create a new image with double width
                combined = Image.new('RGB', (background.width * 2, background.height))
                combined.paste(pred_img, (0, 0))
                combined.paste(gt_img, (background.width, 0))
                combined_images.append(combined)
            
            comparison_path = os.path.join(save_dir, f"sample_{i+1}_comparison.gif")
            create_gif(combined_images, comparison_path)
            
            
import os
import uuid
import numpy as np
from PIL import Image

import os
import uuid
import torch
import numpy as np
from PIL import Image

def visualize_vit_predictions(input_sequence, predictions, save_dir='model_gifs/vit_outputs', background_image_path=None):
    """
    Save input and predicted frames as GIFs, optionally overlaid on a background.

    Args:
        input_sequence (Tensor): Shape [B, T, C, H, W] or [T, C, H, W]
        predictions (Tensor): Shape [B, T, C, H, W] or [T, C, H, W]
        save_dir (str): Output directory for saving images
        background_image_path (str): Optional path to a background image

    Returns:
        dict: {
            'input_gif_path': ...,
            'prediction_gif_path': ...
        }
    """
    os.makedirs(save_dir, exist_ok=True)

    if input_sequence.ndim == 4:
        input_sequence = input_sequence.unsqueeze(0)
    if predictions.ndim == 4:
        predictions = predictions.unsqueeze(0)

    background = None
    if background_image_path and os.path.exists(background_image_path):
        background = Image.open(background_image_path).convert('RGB').resize((128, 128))

    for b in range(input_sequence.shape[0]):
        input_frames = []
        pred_frames = []

        for t in range(input_sequence.shape[1]):
            # Input Frame
            frame = input_sequence[b, t].cpu()
            frame = frame * 0.5 + 0.5
            np_img = frame.squeeze().numpy() * 255
            img = Image.fromarray(np_img.astype(np.uint8)).convert('L').resize((128, 128))
            input_frames.append(img)

        for t in range(predictions.shape[1]):
            frame = predictions[b, t].cpu()
            frame = frame * 0.5 + 0.5
            np_img = frame.squeeze().numpy() * 255
            img = Image.fromarray(np_img.astype(np.uint8)).convert('L').resize((128, 128))
            img=improve_image_quality(img)

            if background:
                img = img.convert("RGBA")
                bg = background.copy().convert("RGBA")
                bg.paste(img, (0, 0), img)
                img = bg.convert("RGB")

            pred_frames.append(img)

        input_gif_name = f"input_{uuid.uuid4().hex[:6]}.gif"
        pred_gif_name = f"pred_{uuid.uuid4().hex[:6]}.gif"
        input_gif_path = os.path.join(save_dir, input_gif_name)
        pred_gif_path = os.path.join(save_dir, pred_gif_name)

        # Save GIFs
        input_frames[0].save(
            input_gif_path,
            save_all=True,
            append_images=input_frames[1:],
            duration=300,  # duration between frames in milliseconds
            loop=0
        )

        pred_frames[0].save(
            pred_gif_path,
            save_all=True,
            append_images=pred_frames[1:],
            duration=300,
            loop=0
        )

        return {
            "input_gif_path": input_gif_path,
            "prediction_gif_path": pred_gif_path
        }

