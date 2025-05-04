from django.urls import path,include
from . import views

urlpatterns = [
    path('waterlevel/', include('Waterlevel.urls', namespace='waterlevel')),
    path('', views.homepage, name='homepage'),  # '' means the root URL
]