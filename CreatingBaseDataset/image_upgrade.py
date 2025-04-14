import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import time
import threading

# Fonction pour afficher le temps écoulé toutes les 10 secondes
def print_elapsed_time(start_time, method_name, stop_event):
    elapsed = 0
    while not stop_event.is_set():
        time.sleep(10)
        elapsed += 10
        print(f"{method_name} - Temps écoulé: {elapsed} secondes")

def upscale_bilinear(img_array, scale_factor=3):
    """
    Méthode 1: Interpolation bilinéaire
    Upscale une image en utilisant l'interpolation bilinéaire avec les pixels voisins
    """
    # Check if image is grayscale or color
    is_grayscale = len(img_array.shape) == 2
    
    # Get dimensions
    if is_grayscale:
        height, width = img_array.shape
        channels = 1
    else:
        height, width, channels = img_array.shape
    
    # Create new image dimensions
    new_height, new_width = height * scale_factor, width * scale_factor
    
    # Initialize upscaled image array
    if is_grayscale:
        upscaled_array = np.zeros((new_height, new_width), dtype=img_array.dtype)
    else:
        upscaled_array = np.zeros((new_height, new_width, channels), dtype=img_array.dtype)
    
    # Utiliser une vraie interpolation bilinéaire
    for i in range(new_height):
        for j in range(new_width):
            # Calculer les coordonnées correspondantes dans l'image originale
            x = j / scale_factor
            y = i / scale_factor
            
            # Trouver les quatre pixels voisins les plus proches
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)
            
            # Calculer les poids pour l'interpolation
            wx = x - x0
            wy = y - y0
            
            if is_grayscale:
                # Interpolation bilinéaire
                pixel = (1 - wx) * (1 - wy) * img_array[y0, x0] + \
                        wx * (1 - wy) * img_array[y0, x1] + \
                        (1 - wx) * wy * img_array[y1, x0] + \
                        wx * wy * img_array[y1, x1]
                
                upscaled_array[i, j] = pixel
            else:
                for c in range(channels):
                    # Interpolation bilinéaire pour chaque canal
                    pixel = (1 - wx) * (1 - wy) * img_array[y0, x0, c] + \
                            wx * (1 - wy) * img_array[y0, x1, c] + \
                            (1 - wx) * wy * img_array[y1, x0, c] + \
                            wx * wy * img_array[y1, x1, c]
                    
                    upscaled_array[i, j, c] = pixel
    
    return upscaled_array

def upscale_gradient(img_array, scale_factor=3):
    """
    Méthode 2: Gradient directionnel
    Upscale une image en utilisant les gradients directionnels
    """
    # Check if image is grayscale or color
    is_grayscale = len(img_array.shape) == 2
    
    # Get dimensions
    if is_grayscale:
        height, width = img_array.shape
        channels = 1
    else:
        height, width, channels = img_array.shape
    
    # Create new image dimensions
    new_height, new_width = height * scale_factor, width * scale_factor
    
    # Initialize upscaled image array
    if is_grayscale:
        upscaled_array = np.zeros((new_height, new_width), dtype=img_array.dtype)
    else:
        upscaled_array = np.zeros((new_height, new_width, channels), dtype=img_array.dtype)
    
    # Utiliser une vraie interpolation basée sur les gradients
    for i in range(new_height):
        for j in range(new_width):
            # Calculer les coordonnées correspondantes dans l'image originale
            x = j / scale_factor
            y = i / scale_factor
            
            # Trouver les quatre pixels voisins les plus proches
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)
            
            # Calculer les poids pour l'interpolation
            wx = x - x0
            wy = y - y0
            
            if is_grayscale:
                # Calculer les gradients
                if x0 > 0 and x1 < width - 1:
                    grad_x = (float(img_array[y0, x1]) - float(img_array[y0, x0-1])) / 2
                else:
                    grad_x = float(img_array[y0, x1]) - float(img_array[y0, x0])
                
                if y0 > 0 and y1 < height - 1:
                    grad_y = (float(img_array[y1, x0]) - float(img_array[y0-1, x0])) / 2
                else:
                    grad_y = float(img_array[y1, x0]) - float(img_array[y0, x0])
                
                # Interpolation avec gradients
                pixel = float(img_array[y0, x0]) + wx * grad_x + wy * grad_y
                
                # Clip values to valid range
                max_val = 255 if img_array.max() > 1 else 1
                pixel = np.clip(pixel, 0, max_val)
                
                upscaled_array[i, j] = pixel
            else:
                for c in range(channels):
                    # Calculer les gradients pour chaque canal
                    if x0 > 0 and x1 < width - 1:
                        grad_x = (float(img_array[y0, x1, c]) - float(img_array[y0, x0-1, c])) / 2
                    else:
                        grad_x = float(img_array[y0, x1, c]) - float(img_array[y0, x0, c])
                    
                    if y0 > 0 and y1 < height - 1:
                        grad_y = (float(img_array[y1, x0, c]) - float(img_array[y0-1, x0, c])) / 2
                    else:
                        grad_y = float(img_array[y1, x0, c]) - float(img_array[y0, x0, c])
                    
                    # Interpolation avec gradients
                    pixel = float(img_array[y0, x0, c]) + wx * grad_x + wy * grad_y
                    
                    # Clip values to valid range
                    max_val = 255 if img_array.max() > 1 else 1
                    pixel = np.clip(pixel, 0, max_val)
                    
                    upscaled_array[i, j, c] = pixel
    
    return upscaled_array

def upscale_spline(img_array, scale_factor=3):
    """
    Méthode 3: Splines cubiques simplifiées
    Upscale une image en utilisant une approximation de splines cubiques
    """
    # Check if image is grayscale or color
    is_grayscale = len(img_array.shape) == 2
    
    # Get dimensions
    if is_grayscale:
        height, width = img_array.shape
        channels = 1
    else:
        height, width, channels = img_array.shape
    
    # Create new image dimensions
    new_height, new_width = height * scale_factor, width * scale_factor
    
    # Initialize upscaled image array
    if is_grayscale:
        upscaled_array = np.zeros((new_height, new_width), dtype=img_array.dtype)
    else:
        upscaled_array = np.zeros((new_height, new_width, channels), dtype=img_array.dtype)
    
    # Utiliser une approximation de spline cubique
    for i in range(new_height):
        for j in range(new_width):
            # Calculer les coordonnées correspondantes dans l'image originale
            x = j / scale_factor
            y = i / scale_factor
            
            # Trouver les quatre pixels voisins les plus proches
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            
            # Calculer les poids pour l'interpolation
            tx = x - x0
            ty = y - y0
            
            # Coefficients de spline cubique
            cx = np.array([
                -0.5 * tx**3 + tx**2 - 0.5 * tx,
                1.5 * tx**3 - 2.5 * tx**2 + 1,
                -1.5 * tx**3 + 2 * tx**2 + 0.5 * tx,
                0.5 * tx**3 - 0.5 * tx**2
            ])
            
            cy = np.array([
                -0.5 * ty**3 + ty**2 - 0.5 * ty,
                1.5 * ty**3 - 2.5 * ty**2 + 1,
                -1.5 * ty**3 + 2 * ty**2 + 0.5 * ty,
                0.5 * ty**3 - 0.5 * ty**2
            ])
            
            if is_grayscale:
                pixel = 0
                for ky in range(4):
                    y_idx = max(0, min(height - 1, y0 + ky - 1))
                    for kx in range(4):
                        x_idx = max(0, min(width - 1, x0 + kx - 1))
                        pixel += img_array[y_idx, x_idx] * cx[kx] * cy[ky]
                
                # Clip values to valid range
                max_val = 255 if img_array.max() > 1 else 1
                pixel = np.clip(pixel, 0, max_val)
                
                upscaled_array[i, j] = pixel
            else:
                for c in range(channels):
                    pixel = 0
                    for ky in range(4):
                        y_idx = max(0, min(height - 1, y0 + ky - 1))
                        for kx in range(4):
                            x_idx = max(0, min(width - 1, x0 + kx - 1))
                            pixel += img_array[y_idx, x_idx, c] * cx[kx] * cy[ky]
                    
                    # Clip values to valid range
                    max_val = 255 if img_array.max() > 1 else 1
                    pixel = np.clip(pixel, 0, max_val)
                    
                    upscaled_array[i, j, c] = pixel
    
    return upscaled_array

def upscale_texture(img_array, scale_factor=3):
    """
    Méthode 4: Méthode basée sur la texture locale
    Upscale une image en préservant la texture locale
    """
    # Check if image is grayscale or color
    is_grayscale = len(img_array.shape) == 2
    
    # Get dimensions
    if is_grayscale:
        height, width = img_array.shape
        channels = 1
    else:
        height, width, channels = img_array.shape
    
    # Create new image dimensions
    new_height, new_width = height * scale_factor, width * scale_factor
    
    # Initialize upscaled image array
    if is_grayscale:
        upscaled_array = np.zeros((new_height, new_width), dtype=img_array.dtype)
    else:
        upscaled_array = np.zeros((new_height, new_width, channels), dtype=img_array.dtype)
    
    # Utiliser une méthode basée sur la texture
    # Calculer d'abord une version bilinéaire
    bilinear = np.zeros_like(upscaled_array)
    
    for i in range(new_height):
        for j in range(new_width):
            # Calculer les coordonnées correspondantes dans l'image originale
            x = j / scale_factor
            y = i / scale_factor
            
            # Trouver les quatre pixels voisins les plus proches
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)
            
            # Calculer les poids pour l'interpolation
            wx = x - x0
            wy = y - y0
            
            if is_grayscale:
                # Interpolation bilinéaire
                pixel = (1 - wx) * (1 - wy) * img_array[y0, x0] + \
                        wx * (1 - wy) * img_array[y0, x1] + \
                        (1 - wx) * wy * img_array[y1, x0] + \
                        wx * wy * img_array[y1, x1]
                
                bilinear[i, j] = pixel
            else:
                for c in range(channels):
                    # Interpolation bilinéaire pour chaque canal
                    pixel = (1 - wx) * (1 - wy) * img_array[y0, x0, c] + \
                            wx * (1 - wy) * img_array[y0, x1, c] + \
                            (1 - wx) * wy * img_array[y1, x0, c] + \
                            wx * wy * img_array[y1, x1, c]
                    
                    bilinear[i, j, c] = pixel
    
    # Appliquer un rehaussement de texture
    kernel_size = 3
    padding = kernel_size // 2
    
    # Pad the bilinear image
    if is_grayscale:
        padded = np.pad(bilinear, padding, mode='edge')
    else:
        padded = np.pad(bilinear, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    
    # Texture enhancement factor
    alpha = 1.5
    
    for i in range(new_height):
        for j in range(new_width):
            if is_grayscale:
                # Extraire le voisinage
                neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
                
                # Calculer la variance locale (mesure de texture)
                local_mean = np.mean(neighborhood)
                local_var = np.var(neighborhood)
                
                # Rehausser la texture en fonction de la variance locale
                texture_factor = 1.0 + alpha * (local_var / (local_var + 0.1))
                
                # Appliquer le rehaussement
                upscaled_array[i, j] = np.clip(bilinear[i, j] * texture_factor, 0, 255 if bilinear.max() > 1 else 1)
            else:
                for c in range(channels):
                    # Extraire le voisinage pour chaque canal
                    neighborhood = padded[i:i+kernel_size, j:j+kernel_size, c]
                    
                    # Calculer la variance locale (mesure de texture)
                    local_mean = np.mean(neighborhood)
                    local_var = np.var(neighborhood)
                    
                    # Rehausser la texture en fonction de la variance locale
                    texture_factor = 1.0 + alpha * (local_var / (local_var + 0.1))
                    
                    # Appliquer le rehaussement
                    upscaled_array[i, j, c] = np.clip(bilinear[i, j, c] * texture_factor, 0, 255 if bilinear.max() > 1 else 1)
    
    return upscaled_array

    
def main():
    # Spécifier directement le chemin d'entrée ici
    input_path = "e:\\kraya_espri_3eme\\Sem2\\projet\\GitHub\\Water-Scarcity\\visualization_output\\ESoil_tavg\\A20000101.png"
    
    # Créer automatiquement les chemins de sortie basés sur le chemin d'entrée
    filename, ext = os.path.splitext(input_path)
    output_bilinear = f"{filename}_bilinear{ext}"
    output_gradient = f"{filename}_gradient{ext}"
    output_spline = f"{filename}_spline{ext}"
    output_texture = f"{filename}_texture{ext}"
    
    # Default scale factor is 3
    scale_factor = 3
    
    # Vous pouvez toujours utiliser les arguments de ligne de commande si fournis
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
        filename, ext = os.path.splitext(input_path)
        output_bilinear = f"{filename}_bilinear{ext}"
        output_gradient = f"{filename}_gradient{ext}"
        output_spline = f"{filename}_spline{ext}"
        output_texture = f"{filename}_texture{ext}"
    
    if len(sys.argv) >= 3:
        try:
            scale_factor = int(sys.argv[2])
        except ValueError:
            print("Scale factor must be an integer. Using default value of 3.")
    
    # Charger l'image
    try:
        img = Image.open(input_path)
        img_array = np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Dictionnaire pour stocker les temps d'exécution
    execution_times = {}
    
    # Appliquer les quatre méthodes d'upscaling avec suivi du temps
    print("Applying bilinear interpolation...")
    stop_event = threading.Event()
    timer_thread = threading.Thread(target=print_elapsed_time, args=(time.time(), "Bilinear", stop_event))
    timer_thread.daemon = True
    timer_thread.start()
    
    start_time = time.time()
    bilinear_array = upscale_bilinear(img_array, scale_factor)
    bilinear_time = time.time() - start_time
    execution_times["Bilinear"] = bilinear_time
    
    stop_event.set()
    bilinear_img = Image.fromarray(bilinear_array if len(img_array.shape) == 2 else bilinear_array.astype(np.uint8))
    bilinear_img.save(output_bilinear)
    print(f"Bilinear method completed in {bilinear_time:.2f} seconds")
    
    print("Applying gradient method...")
    stop_event = threading.Event()
    timer_thread = threading.Thread(target=print_elapsed_time, args=(time.time(), "Gradient", stop_event))
    timer_thread.daemon = True
    timer_thread.start()
    
    start_time = time.time()
    gradient_array = upscale_gradient(img_array, scale_factor)
    gradient_time = time.time() - start_time
    execution_times["Gradient"] = gradient_time
    
    stop_event.set()
    gradient_img = Image.fromarray(gradient_array if len(img_array.shape) == 2 else gradient_array.astype(np.uint8))
    gradient_img.save(output_gradient)
    print(f"Gradient method completed in {gradient_time:.2f} seconds")
    
    print("Applying spline method...")
    stop_event = threading.Event()
    timer_thread = threading.Thread(target=print_elapsed_time, args=(time.time(), "Spline", stop_event))
    timer_thread.daemon = True
    timer_thread.start()
    
    start_time = time.time()
    spline_array = upscale_spline(img_array, scale_factor)
    spline_time = time.time() - start_time
    execution_times["Spline"] = spline_time
    
    stop_event.set()
    spline_img = Image.fromarray(spline_array if len(img_array.shape) == 2 else spline_array.astype(np.uint8))
    spline_img.save(output_spline)
    print(f"Spline method completed in {spline_time:.2f} seconds")
    
    print("Applying texture method...")
    stop_event = threading.Event()
    timer_thread = threading.Thread(target=print_elapsed_time, args=(time.time(), "Texture", stop_event))
    timer_thread.daemon = True
    timer_thread.start()
    
    start_time = time.time()
    texture_array = upscale_texture(img_array, scale_factor)
    texture_time = time.time() - start_time
    execution_times["Texture"] = texture_time
    
    stop_event.set()
    texture_img = Image.fromarray(texture_array if len(img_array.shape) == 2 else texture_array.astype(np.uint8))
    texture_img.save(output_texture)
    print(f"Texture method completed in {texture_time:.2f} seconds")
    
    # Afficher les résultats
    try:
        original = np.array(img)
        
        plt.figure(figsize=(15, 10))
        
        # Image originale
        plt.subplot(2, 3, 1)
        plt.imshow(original)
        plt.title(f'Original Image ({original.shape[1]}x{original.shape[0]})')
        plt.axis('off')
        
        # Méthode 1: Bilinéaire
        plt.subplot(2, 3, 2)
        plt.imshow(bilinear_array if len(img_array.shape) == 2 else bilinear_array.astype(np.uint8))
        plt.title(f'Bilinear Method - {bilinear_time:.2f}s\n({bilinear_array.shape[1]}x{bilinear_array.shape[0]})')
        plt.axis('off')
        
        # Méthode 2: Gradient
        plt.subplot(2, 3, 3)
        plt.imshow(gradient_array if len(img_array.shape) == 2 else gradient_array.astype(np.uint8))
        plt.title(f'Gradient Method - {gradient_time:.2f}s\n({gradient_array.shape[1]}x{gradient_array.shape[0]})')
        plt.axis('off')
        
        # Méthode 3: Spline
        plt.subplot(2, 3, 5)
        plt.imshow(spline_array if len(img_array.shape) == 2 else spline_array.astype(np.uint8))
        plt.title(f'Spline Method - {spline_time:.2f}s\n({spline_array.shape[1]}x{spline_array.shape[0]})')
        plt.axis('off')
        
        # Méthode 4: Texture
        plt.subplot(2, 3, 6)
        plt.imshow(texture_array if len(img_array.shape) == 2 else texture_array.astype(np.uint8))
        plt.title(f'Texture Method - {texture_time:.2f}s\n({texture_array.shape[1]}x{texture_array.shape[0]})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Original image size: {original.shape[1]}x{original.shape[0]}")
        print(f"Upscaled image size: {bilinear_array.shape[1]}x{bilinear_array.shape[0]}")
        print(f"Images saved to:")
        print(f"  - {output_bilinear}")
        print(f"  - {output_gradient}")
        print(f"  - {output_spline}")
        print(f"  - {output_texture}")
        
        # Afficher un résumé des temps d'exécution
        print("\nRésumé des temps d'exécution:")
        for method, exec_time in execution_times.items():
            print(f"{method}: {exec_time:.2f} secondes")
        
    except Exception as e:
        print(f"Error displaying images: {e}")

if __name__ == "__main__":
    main()