from django.urls import path,include
from . import views
from django.conf.urls.static import static # Required for static files
from django.conf import settings # Required for static files

urlpatterns = [
    path('waterlevel/', include('Waterlevel.urls', namespace='waterlevel')),
    path('', views.homepage, name='homepage'),
    path('chat_api/', include('chat.urls')),
     path('irrig/', include('Irrig.urls')), 
    path('watershed/' , include('Watershed.urls', namespace='watershed')),
    path('drought/', include('Drought.urls', namespace='drought')),

      # '' means the root URL
]

# Servir les fichiers statiques et médias en développement
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)