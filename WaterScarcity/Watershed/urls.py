from django.urls import path
from . import views

app_name = 'watershed'

urlpatterns = [
    path('predict-deforestation/', views.predict_watershed, name='predict_watershed'),
]
