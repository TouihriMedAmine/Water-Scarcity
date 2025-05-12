from django.urls import path
from . import views

app_name = 'runoff'

urlpatterns = [
    path('predict_runoff', views.predict_runoff, name='predict_runoff'),
]