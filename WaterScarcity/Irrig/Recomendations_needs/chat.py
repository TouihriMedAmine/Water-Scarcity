from langchain_ollama import OllamaLLM as Ollama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import sys
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_dir = os.path.join(parent_dir, "Recomendations_needs")
sys.path.insert(0, model_dir)

# Load embeddings and vectorstore_reco
embeddings = OllamaEmbeddings(model="mistral")
vectorstore_reco = FAISS.load_local("Irrig/Recomendations_needs/vectorstore_reco/", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore_reco.as_retriever(search_kwargs={"k": 3})


try:
    from Joined_EndPoint import get_all_predictions
except ImportError as e:
    print(f"Erreur lors de l'importation de Joined_EndPoint: {e}")
    print(f"Veuillez vérifier que le fichier Joined_EndPoint.py se trouve bien dans {model_dir}")
    # Vous pouvez choisir de quitter le script ici ou de continuer sans cette fonctionnalité
    # sys.exit(1) 
    get_all_predictions = None # Pour éviter des erreurs si l'import échoue mais que le script continue


# Initialize LLM
llm = Ollama(
    model="mistral",
    temperature=0.7,
    top_p=0.9,
    top_k=50
)


def get_response(user_query):
    if not user_query.strip():
        return "Please enter a valid question."

    try:
        language = detect(user_query)
    except LangDetectException:
        language = "en"  # default to English if detection fails

    # Retrieve relevant documents
    context = retriever.invoke(user_query)
    context_text = "\n".join([doc.page_content for doc in context])

    # Create language-specific instructions
    persona_instructions = """
You are "AgriConsult", an AI assistant expert in agricultural adaptation to climate challenges. Your mission is to provide personalized recommendations for crop types and adaptation strategies based on specific environmental conditions (temperature, precipitation, soil health) and available contextual information.

When the user provides data such as:
- Average Temperature (°C)
- Total Precipitation (mm)
- Soil Health Index (or similar like Soil Moisture)

And asks for agricultural advice, use the provided context (documents retrieved via vector search) and your general knowledge to generate a concise report. This report should ideally include:
1.  **Suggested Crop Type(s):** Suitable for the provided conditions and informed by the context.
2.  **Recommended Adaptation Strategies:** Relevant to the suggested crops and conditions, drawing from the context.
3.  **Possible Alternatives:** If the context suggests other viable crops or strategies for similar conditions.
4.  **General Steps to Follow:** Key actions the farmer might consider.

Behavior Guide:
- Be informative, precise, and practical.
- If information comes from the consulted documents (context), you can mention it (e.g., "According to the analyzed documents...").
- Stay focused on agricultural recommendations related to climate, soil, crop types, and adaptation strategies.
- Never make up facts. If unsure or if the context doesn't provide enough information, state it clearly.
- Aim for a structured and easy-to-understand response.
- Maximum response length: approximately 150-200 words to allow for a succinct yet informative report.
"""

    prompt = f"""
{persona_instructions}

Use the following context to answer the user's question:

--- 
📚 Context: 
{context_text}
---
❓User Question: 
{user_query}
---
Your Response:
"""

    return llm.invoke(prompt).strip()


def format_input_conditions_display(predictions: dict) -> str:
    """
    Formats the prediction data into a display string in English.
    """
    display_parts = ["Input Conditions:"]
    if "AvgSurfT" in predictions and "prediction" in predictions["AvgSurfT"]:
        display_parts.append(f"- Average Surface Temperature: {predictions['AvgSurfT']['prediction']} {predictions['AvgSurfT']['unit']}")
    if "Rainf" in predictions and "prediction" in predictions["Rainf"]:
        display_parts.append(f"- Total Precipitation: {predictions['Rainf']['prediction']} {predictions['Rainf']['unit']}")
    if "PotEvap" in predictions and "prediction" in predictions["PotEvap"]:
        display_parts.append(f"- Potential Evapotranspiration: {predictions['PotEvap']['prediction']} {predictions['PotEvap']['unit']}")
    if "SoilM" in predictions and "prediction" in predictions["SoilM"]:
        display_parts.append(f"- Soil Moisture (0-100cm): {predictions['SoilM']['prediction']} {predictions['SoilM']['unit']}")
    # Vous pouvez ajouter d'autres prédictions ici si elles sont pertinentes pour l'affichage
    # Exemple:
    # if "OtherParam" in predictions and "prediction" in predictions["OtherParam"]:
    #     display_parts.append(f"- Other Parameter: {predictions['OtherParam']['prediction']} {predictions['OtherParam']['unit']}")
    
    if len(display_parts) == 1: # Only "Input Conditions:"
        return "No specific input conditions data available to display."
        
    return "\n".join(display_parts)

def generate_llm_query_from_predictions(predictions: dict) -> str:
    """
    Generates an English query for the LLM based on prediction data.
    """
    query_parts = []
    if "AvgSurfT" in predictions and "prediction" in predictions["AvgSurfT"]:
        query_parts.append(f"Average Surface Temperature: {predictions['AvgSurfT']['prediction']} {predictions['AvgSurfT']['unit']}")
    if "Rainf" in predictions and "prediction" in predictions["Rainf"]:
        query_parts.append(f"Total Precipitation: {predictions['Rainf']['prediction']} {predictions['Rainf']['unit']}")
    if "PotEvap" in predictions and "prediction" in predictions["PotEvap"]:
        query_parts.append(f"Potential Evapotranspiration: {predictions['PotEvap']['prediction']} {predictions['PotEvap']['unit']}")
    if "SoilM" in predictions and "prediction" in predictions["SoilM"]:
        query_parts.append(f"Soil Moisture (0-100cm): {predictions['SoilM']['prediction']} {predictions['SoilM']['unit']}")
    # Assurez-vous que les paramètres ici correspondent à ceux que le LLM doit considérer
    
    if not query_parts:
        return "No prediction data available to form a query."
        
    return ", ".join(query_parts) + ". What crops and adaptation strategies do you recommend based on these conditions?"

def get_agricultural_advice_with_predictions(longitude: str, latitude: str, date_str: str):
    """
    Obtient des conseils agricoles en utilisant les prédictions de modèles et le LLM.
    La sortie sera en anglais, avec les conditions d'entrée affichées en premier.
    """
    if get_all_predictions is None:
        return "Prediction functionality is unavailable due to an import error." # Message en anglais

    print(f"Fetching predictions for lon={longitude}, lat={latitude}, date={date_str}...")
    predictions_data = get_all_predictions(date_str=date_str, lon_str=str(longitude), lat_str=str(latitude))

    if "error" in predictions_data: # Si get_all_predictions retourne une erreur globale
        return f"Error fetching predictions: {predictions_data['error']}" # Message en anglais
    
    valid_predictions = {k: v for k, v in predictions_data.items() if isinstance(v, dict) and "error" not in v}
    
    if not valid_predictions:
         error_messages_list = []
         for k, v in predictions_data.items():
             if isinstance(v, dict) and "error" in v:
                 error_messages_list.append(f"{k}: {v['error']}")
             elif not isinstance(v,dict): # Cas où une prédiction n'est pas un dictionnaire attendu
                 error_messages_list.append(f"{k}: Unexpected data format")

         if not error_messages_list: # Si predictions_data était vide ou n'avait pas le format attendu
             return "No valid prediction data was processed."

         error_messages_str = "; ".join(error_messages_list)
         return f"All predictions failed or data is invalid. Errors: {error_messages_str}" # Message en anglais


    print(f"Predictions obtained: {valid_predictions}")
    
    # 1. Formater les conditions d'entrée pour l'affichage (en anglais)
    conditions_display_string = format_input_conditions_display(valid_predictions)
    
    # 2. Générer la requête pour le LLM (en anglais)
    llm_query = generate_llm_query_from_predictions(valid_predictions)
    print(f"Constructed query for LLM: {llm_query}")

    if llm_query == "No prediction data available to form a query.":
        # Retourner la chaîne des conditions même si la requête LLM ne peut être formée
        return f"{conditions_display_string}\n\nCannot formulate a recommendation as essential prediction data is missing."

    # 3. Obtenir la réponse du LLM (qui sera en anglais)
    llm_recommendation = get_response(llm_query)

    # 4. Combiner la chaîne d'affichage des conditions et la recommandation du LLM
    final_output = f"{conditions_display_string}\n\nAgricultural Recommendation:\n{llm_recommendation}"
    
    return final_output


# Basic CLI
def chat():
    print("\n🌱 AgriConsult Chatbot - Type 'exit' to quit.")
    print("Ask your question including average temperature, total precipitation, and soil health index.")
    print("Example: Skin Tempeture: 280 K, Precipitation: 300mm, Soil Health Index: 0.7. What crops and strategies do you recommend?")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower().strip() == "exit":
                print("👋 Goodbye!")
                break
            response = get_response(user_input)
            print("Bot:", response)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break