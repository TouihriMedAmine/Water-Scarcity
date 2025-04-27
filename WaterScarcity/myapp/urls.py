from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),  # '' means the root URL
]