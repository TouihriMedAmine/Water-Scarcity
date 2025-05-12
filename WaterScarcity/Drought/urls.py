from django.urls import path
from . import views

app_name = 'drought'

urlpatterns = [
    
    path('predict_drought/', views.predict_drought, name='predict_drought'),
]
