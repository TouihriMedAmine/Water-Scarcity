from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt # Pour simplifier les tests AJAX, mais attention en production
import sys
import os

# Ajouter le chemin vers le dossier Models_KERAS au sys.path
# Attention: Utiliser des chemins absolus ou relatifs robustes est préférable
keras_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Models_KERAS'))
if keras_models_path not in sys.path:
    sys.path.append(keras_models_path)

# Importer la fonction spécifique après avoir ajusté le path
try:
    from Fonctions_Integration import predict_uploaded_image
except ImportError as e:
    # Gérer l'erreur si l'import échoue
    print(f"Erreur d'importation: {e}")
    # Définir une fonction factice ou lever une exception pour indiquer le problème
    def predict_uploaded_image(file):
        raise ImportError("Impossible d'importer predict_uploaded_image depuis Models_KERAS.Fonctions_Integration")

# Vue pour gérer la requête POST du formulaire d'estimation
@csrf_exempt # Important pour les requêtes AJAX POST sans configuration CSRF complexe côté JS
             # En production, gérez le token CSRF correctement avec JavaScript
def estimate_water_level_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        try:
            # Appeler votre fonction de prédiction
            result = predict_uploaded_image(uploaded_file)
            # Retourner le résultat en JSON
            return JsonResponse({
                'success': True,
                'class': result.get('class', 'Inconnue'),
                'predicted_depth': result.get('predicted_depth', 'Erreur'),
                'blue_intensity': result.get('blue_intensity', 'N/A')
            })
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    # Gérer les cas où ce n'est pas une requête POST valide
    return JsonResponse({'success': False, 'error': 'Requête invalide ou image manquante'}, status=400)

# Ajoutez ici la vue qui affiche la page d'accueil si elle est gérée par cette app
# def homepage_view(request):
#     return render(request, 'Homepage/homepage.html')