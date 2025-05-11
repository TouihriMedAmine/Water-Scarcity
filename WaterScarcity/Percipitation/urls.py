# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('percip/', views.predict_view2, name='predict'),
    path('trend/', views.predict_trend_view, name='trend'),
    path('percipt/', views.vit_prediction_view, name='percipt'),	
    # ... other URLs
]