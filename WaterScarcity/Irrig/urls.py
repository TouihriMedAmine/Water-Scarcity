from django.urls import path
from . import views

app_name = 'Irrig' # Optionnel, mais utile pour les espaces de noms

urlpatterns = [
    path('irrigation-recommendations/', views.irrigation_recommendation_view, name='irrigation_recommendation'),
]