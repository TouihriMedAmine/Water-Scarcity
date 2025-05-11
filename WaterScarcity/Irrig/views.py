from django.shortcuts import render
from django.http import HttpRequest, JsonResponse # Importez JsonResponse
from Irrig.Recomendations_needs.chat import get_agricultural_advice_with_predictions

def irrigation_recommendation_view(request: HttpRequest):
    if request.method == 'POST':
        longitude = request.POST.get('longitude')
        latitude = request.POST.get('latitude')
        date_str = request.POST.get('date')

        response_data = {} # Pour stocker les données à renvoyer en JSON

        if longitude and latitude and date_str:
            try:
                recommendations = get_agricultural_advice_with_predictions(
                    longitude=str(longitude),
                    latitude=str(latitude),
                    date_str=str(date_str)
                )
                # Vérifiez si la recommandation contient un message d'erreur connu
                # ou si c'est une recommandation valide.
                # Adaptez cette logique si votre fonction get_agricultural_advice_with_predictions
                # peut retourner des erreurs de manière plus structurée.
                if "erreur" in recommendations.lower() or "error" in recommendations.lower() or "veuillez fournir" in recommendations.lower():
                    response_data['error_message'] = recommendations
                else:
                    response_data['recommendations'] = recommendations
            except Exception as e:
                response_data['error_message'] = f"Une erreur interne s'est produite lors de la récupération des recommandations : {str(e)}"
        else:
            response_data['error_message'] = "Veuillez fournir la longitude, la latitude et la date."

        # Si la requête est une requête AJAX, renvoyer une JsonResponse
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            if 'error_message' in response_data:
                return JsonResponse(response_data, status=400) # Erreur client ou serveur
            return JsonResponse(response_data)
        else:
            # Comportement pour une soumission non-AJAX (si nécessaire, sinon peut être supprimé)
            # Ici, nous allons simplement passer le contexte au template comme avant,
            # mais idéalement, pour une API AJAX, cette branche ne serait pas souvent utilisée.
            context = response_data
            return render(request, 'Homepage/homepage.html', context)

    # Pour les requêtes GET ou autres méthodes non POST
    # Si c'est une requête AJAX GET, on pourrait retourner une erreur ou des données par défaut
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'error_message': 'Méthode GET non supportée pour les recommandations via AJAX.'}, status=405)
    
    # Comportement par défaut pour une requête GET (afficher la page)
    return render(request, 'Homepage/homepage.html', {})