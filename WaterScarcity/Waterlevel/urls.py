from django.urls import path,include
from . import views # Importer les vues de l'application Waterlevel

app_name = 'waterlevel' # Nom de l'application pour les namespaces d'URL

urlpatterns = [
    # URL pour la vue d'estimation
    path('estimate/', views.estimate_water_level_view, name='estimate_water_level'),
    # Ajoutez d'autres URLs pour cette application si nécessaire
]