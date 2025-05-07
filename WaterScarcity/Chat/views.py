from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
import logging
from .chat import get_response as get_bot_response_from_logic

@csrf_exempt
def get_chat_api_response(request):
    if request.method == 'POST':
        try:
            user_message = request.POST.get('message')
            if not user_message:
                data = json.loads(request.body)
                user_message = data.get('message')

            if user_message:
                # Appeler votre logique de chatbot
                # Notez qu'on ne passe plus VECTORSTORE_PATH ici
                bot_response_text = get_bot_response_from_logic(user_message)
                return JsonResponse({'response': bot_response_text})
            else:
                return JsonResponse({'error': 'No message provided'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON in request body or no message in form data'}, status=400)
        except Exception as e:
            # Il est bon de logger l'erreur ici pour le débogage
            logging.error(f"Chatbot error: {e}", exc_info=True) # Décommentez et modifiez cette ligne
            return JsonResponse({'error': f'An internal error occurred: {str(e)}'}, status=500)
    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)