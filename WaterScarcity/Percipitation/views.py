# myapp/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from datetime import date,datetime,timedelta
from PIL import Image
import random
from django.shortcuts import render
import io
from PIL import Image
import torch
import os
import numpy as np
import tempfile
from .model import ConvLSTM
from torchvision import transforms
from .save_images import save_images,visualize_vit_predictions
from .Unet import UNet
from .utils import  extract_trends_by_state, plot_state_trend,run_inference,run_unet
from .STformer import VisionTransformer
import traceback
import pandas as pd

def predict_view(request):
    if request.method == 'POST':
        try:
            # Get uploaded images (as a list)
            uploaded_files = request.FILES.getlist('photos')
            print("Uploaded files:", len(uploaded_files))
            if len(uploaded_files) < 2:
                return render(request, 'Homepage/homepage.html', {'error': 'Please upload at least 2 images.'})

            # Save images temporarily & preprocess
            uploaded_files.sort(key=lambda f: f.name)

            # Step 2: Group into 2 sequences of 4 images
            sequences = []
            for i in range(0, 8, 4):  # i = 0, 4
                sequence_images = []
                for file in uploaded_files[i:i+4]:
                    image = Image.open(file).convert('RGB').resize((256, 256))
                    image = transforms.ToTensor()(image)
                    image = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(image)
                    sequence_images.append(image)
                sequence_tensor = torch.stack(sequence_images)  # Shape: [4, 3, 256, 256]
                sequences.append(sequence_tensor)

            # Step 3: Stack into final input tensor
            input_sequence = torch.stack(sequences)  # Shape: [2, 4, 3, 256, 256]
            print(f"Input sequence shape: {input_sequence.shape}")
            # Dummy target (for visual comparison only)
            target = input_sequence[:, -1]  # Last frame as fake target

            # Load model
            model = ConvLSTM(input_channels=3, hidden_channels=64, kernel_size=3, num_layers=3).to('cpu')
            model.load_state_dict(torch.load('convlstm_model25.pth', map_location='cpu'))
            model.eval()
            all_pred=[]
            # Run prediction
            
            with torch.no_grad():
                predicted = model(input_sequence)      # shape: [1, C, H, W]
            all_pred.append(predicted)
            print(f"Predicted shape: {predicted.shape}")
            predictions = torch.cat(all_pred, dim=1)
            print(input_sequence.shape, target.shape, predicted.shape)
            # Save the result GIF
            
            gif_dir = os.path.join(tempfile.gettempdir(), "./static/model_gifs")
            os.makedirs(gif_dir, exist_ok=True)
            gif_path = os.path.join(gif_dir, "result.gif")
            save_images(predictions, target, batch_idx=0)  # You may need to adapt this to take single samples

            return render(request, 'Homepage/homepage.html', {
                'gif_path': 'model_gifs/comparison_batch0.gif'  # Or use MEDIA_URL + relative path
            })

        except Exception as e:
            return render(request, 'Homepage/homepage.html', {
                'error': str(e)
            })

    return render(request, 'Homepage/homepage.html')


    
def predict_view2(request):
    if request.method == 'POST':
        try:
            uploaded_files = request.FILES.getlist('photos')
            if len(uploaded_files) < 2:
                return render(request, 'Homepage/homepage.html', {'error': 'Please upload at least 2 images.'})

            # Sort files by name assuming filenames are yyyymmddhh
            uploaded_files.sort(key=lambda f: f.name)
            sequences = []
            photo_names = [file.name[:-4] for file in uploaded_files]
            formatted_photo_names = []
            dates = []

            for photo_name in photo_names:
                # Parse the date and hour
                year = int(photo_name[:4])
                month = int(photo_name[4:6])
                day = int(photo_name[6:8])
                hour = int(photo_name[8:10])
                
                # Create a datetime object
                dt = datetime(year, month, day, hour)
                
                # Format and store
                formatted_photo_names.append(dt.strftime('%B %d, %Y at %H:00'))
                dates.append(dt)  # Add the base datetime to list

                # Add 3 additional hours
                for i in range(1, 4):
                    future_dt = dt + timedelta(hours=i)
                    formatted_photo_names.append(future_dt.strftime('%B %d, %Y at %H:00'))
                    dates.append(future_dt)
            print("Uploaded files:", len(uploaded_files))

            target_size = (128, 128)

            for i in range(0, len(uploaded_files), 3):
                sequence_images = []
                for file in uploaded_files[i:i+3]:
                    image = Image.open(file).convert('RGB').resize(target_size)
                    image = transforms.ToTensor()(image)
                    image = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(image)
                    sequence_images.append(image)

                while len(sequence_images) < 3:
                    sequence_images.append(torch.zeros(3, *target_size))

                sequence_tensor = torch.stack(sequence_images)
                sequences.append(sequence_tensor)

            input_sequence = torch.stack(sequences)
            model_checkpoint = "static/modelsP/predrnn25.pth"

            preds,_=run_inference(model_checkpoint, input_sequence, num_samples=len(sequences))
            print(f"Predicted shape: {preds.shape}")
            state_name = request.POST.get('state')
            dates=pd.DatetimeIndex(dates).unique().sort_values()

            if state_name:
                trend,dates,plot_path=run_unet(preds,state_name,dates)
                return render(request, 'Homepage/homepage.html', {
                'gif_path': 'model_gifs/sample_1_predicted.gif',
                'photo_names': formatted_photo_names,
                'plot_path': '/' + plot_path,
                })
            
            return render(request, 'Homepage/homepage.html', {
                'gif_path': 'model_gifs/sample_1_predicted.gif',
                'photo_names': formatted_photo_names,
                
            })
            

        except Exception as e:
            print(f"Error: {str(e)}")
            traceback.print_exc()
            return render(request, 'Homepage/homepage.html', {'error': str(e)})
    
    return render(request, 'Homepage/homepage.html')





def predict_trend_view(request):
    if request.method == 'POST':
        try:
            uploaded_files = request.FILES.getlist('photosUnet')
            state_name = request.POST.get('state')
            print("Uploaded files:", len(uploaded_files))

            if len(uploaded_files) < 2:
                return JsonResponse({
                    'success': False,
                    'error': 'Please upload exactly 8 images.'
                }, status=400)

            uploaded_files.sort(key=lambda f: f.name)

            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])

            input_tensors = []
            image_dates = []

            for file in uploaded_files:
                image = Image.open(file).convert('RGB')
                tensor = transform(image)
                input_tensors.append(tensor)
                image_dates.append(os.path.splitext(file.name)[0])

            batch = torch.stack(input_tensors)  # Shape: [8, 3, 256, 256]

            # Load model
            model = UNet(n_channels=3, n_classes=4)  # Adjust class count if needed
            model.load_state_dict(torch.load('static/modelsP/unet.pth', map_location='cpu'))
            model.eval()

            with torch.no_grad():
                predictions = model(batch)  # [8, C, 256, 256]

            predicted_masks = predictions  # [8, 256, 256]

            # Analyze and plot trends
            trend = extract_trends_by_state(predicted_masks, state_name)
            plot_path = f'static/trends/{state_name}.png'
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plot_state_trend(state_name, trend, image_dates, save_path=plot_path)
            predicted_trend = [np.float32(0.85), np.float32(0.9)]

# Convert to native Python float
            predicted_trend = [float(x) for x in trend]

            print(image_dates)
            return JsonResponse({
                'success': True,
                'state': state_name,
                'trend': predicted_trend,
                'plot_path': '/' + plot_path  # Add slash to make it relative to static/
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

    return JsonResponse({
        'success': False,
        'error': 'Only POST method is allowed.'
    }, status=405)



#STformer prediction view
def vit_prediction_view(request):
    if request.method == 'POST':
        try:
            uploaded_files = request.FILES.getlist('imagestrans')
            if len(uploaded_files) < 2:
                return JsonResponse({
                    'success': False,
                    'error': 'Please upload at least 2 images.'
                }, status=400)

            uploaded_files.sort(key=lambda f: f.name)

            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # assuming grayscale; adjust if RGB
            ])

            tensors = []
            for file in uploaded_files:
                image = Image.open(file).convert('L')  # grayscale for ViT
                tensor = transform(image)
                tensors.append(tensor)

            input_sequence = torch.stack(tensors).unsqueeze(0)  # shape: [1, T, C, H, W]

            # Load your ViT model
            model =VisionTransformer(input_len=4, target_len=4, image_size=(128,128)).to("cpu")
            model.load_state_dict(torch.load('vit_model.pth', map_location='cpu'))
            model.eval()

            with torch.no_grad():
                predictions = model(input_sequence)  # shape: [1, T, C, H, W]

            results = visualize_vit_predictions(input_sequence, predictions,background_image_path="empty_map_us.png")

            return JsonResponse({
                'success': True,
                'input_gif_path': '/' + results['input_gif_path'],
                'prediction_gif_path': '/' + results['prediction_gif_path']
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

    return JsonResponse({
        'success': False,
        'error': 'Only POST method is allowed.'
    }, status=405)