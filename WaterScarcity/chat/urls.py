from django.urls import path
from . import views

urlpatterns = [
    path('get_response/', views.get_chat_api_response, name='get_chat_api_response'),
]