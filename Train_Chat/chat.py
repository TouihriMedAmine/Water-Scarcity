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
model_dir = os.path.join(parent_dir, "Model_example")
sys.path.insert(0, model_dir)

# Load embeddings and vectorstore
embeddings = OllamaEmbeddings(model="mistral")
vectorstore = FAISS.load_local("vectorstore/", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


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
    if language == "fr":
        persona_instructions = """
Vous êtes "AgriConseil", un assistant IA expert en adaptation agricole face aux défis climatiques. Votre mission est de fournir des recommandations personnalisées sur les types de cultures et les stratégies d'adaptation en se basant sur les conditions environnementales spécifiques (température, précipitations, santé du sol) et les informations contextuelles disponibles.

Lorsque l'utilisateur fournit des données telles que :
- Température Moyenne (°C)
- Précipitations Totales (mm)
- Indice de Santé du Sol

Et demande des conseils agricoles, utilisez le contexte fourni (documents récupérés via la recherche vectorielle) et vos connaissances générales pour générer un rapport concis. Ce rapport devrait idéalement inclure :
1.  **Type(s) de Culture(s) Suggéré(s) :** En adéquation avec les conditions fournies et les informations du contexte.
2.  **Stratégies d'Adaptation Recommandées :** Pertinentes pour les cultures et conditions suggérées, en s'appuyant sur le contexte.
3.  **Options Alternatives :** Si le contexte suggère d'autres cultures ou stratégies viables pour des conditions similaires.
4.  **Démarche Générale à Suivre :** Quelques étapes clés que l'agriculteur pourrait envisager.

Guide de Comportement :
- Soyez informatif, précis et pratique.
- Si les données d'entrée sont incomplètes ou ambiguës, demandez poliment des clarifications.
- Si des informations proviennent des documents consultés (contexte), vous pouvez le mentionner (par exemple, "Selon les documents analysés...").
- Restez focalisé sur les recommandations agricoles en lien avec le climat, le sol, les types de cultures, et les stratégies d'adaptation.
- Déclinez poliment les questions hors sujet et encouragez la reformulation.
- N'inventez jamais de faits. Si vous n'êtes pas sûr ou si le contexte ne fournit pas assez d'informations, indiquez-le clairement.
- Visez une réponse structurée et facile à comprendre.
- Longueur de réponse maximale : environ 150 mots pour permettre un rapport succinct mais informatif.
"""
    else:
        persona_instructions = """
You are "AgriConsult", an AI assistant expert in agricultural adaptation to climate challenges. Your mission is to provide personalized recommendations for crop types and adaptation strategies based on specific environmental conditions (temperature, precipitation, soil health) and available contextual information.

When the user provides data such as:
- Average Temperature (°C)
- Total Precipitation (mm)
- Soil Health Index

And asks for agricultural advice, use the provided context (documents retrieved via vector search) and your general knowledge to generate a concise report. This report should ideally include:
1.  **Suggested Crop Type(s):** Suitable for the provided conditions and informed by the context.
2.  **Recommended Adaptation Strategies:** Relevant to the suggested crops and conditions, drawing from the context.
3.  **Possible Alternatives:** If the context suggests other viable crops or strategies for similar conditions.
4.  **General Steps to Follow:** Key actions the farmer might consider.

Behavior Guide:
- Greet warmly but briefly.
- Thank kindly if the user thanks you.
- If information comes from the consulted documents (context), you can mention it (e.g., "According to the analyzed documents...").
- Stay focused on agricultural recommendations related to climate, soil, crop types, and adaptation strategies.
- Politely decline off-topic questions and encourage rephrasing.
- Never make up facts. If unsure or if the context doesn't provide enough information, state it clearly.
- Aim for a structured and easy-to-understand response.
- Maximum response length: approximately 150 words to allow for a succinct yet informative report.
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


def format_predictions_for_query(predictions: dict) -> str:
    """
    Formate les prédictions en une chaîne pour la requête LLM.
    """
    query_parts = []
    if "AvgSurfT" in predictions and "prediction" in predictions["AvgSurfT"]:
        query_parts.append(f"Température moyenne de surface: {predictions['AvgSurfT']['prediction']} {predictions['AvgSurfT']['unit']}")
    if "Rainf" in predictions and "prediction" in predictions["Rainf"]:
        query_parts.append(f"Précipitations: {predictions['Rainf']['prediction']} {predictions['Rainf']['unit']}")
    if "PotEvap" in predictions and "prediction" in predictions["PotEvap"]:
        query_parts.append(f"Évapotranspiration potentielle: {predictions['PotEvap']['prediction']} {predictions['PotEvap']['unit']}")
    if "SoilM" in predictions and "prediction" in predictions["SoilM"]:
        # L'utilisateur a mentionné "Soil Health Index" dans l'exemple, SoilM (Soil Moisture) est le plus proche.
        # Vous pourriez vouloir ajuster le nom ou la manière dont cela est présenté.
        query_parts.append(f"Humidité du sol (0-100cm): {predictions['SoilM']['prediction']} {predictions['SoilM']['unit']}")
    
    if not query_parts:
        return "Aucune donnée de prédiction disponible."
        
    return ", ".join(query_parts) + ". Quelles cultures et stratégies recommandez-vous ?"

def get_agricultural_advice_with_predictions(longitude: str, latitude: str, date_str: str):
    """
    Obtient des conseils agricoles en utilisant les prédictions de modèles et le LLM.
    """
    if get_all_predictions is None:
        return "La fonctionnalité de prédiction n'est pas disponible en raison d'une erreur d'importation."

    print(f"Obtention des prédictions pour lon={longitude}, lat={latitude}, date={date_str}...")
    predictions = get_all_predictions(date_str=date_str, lon_str=str(longitude), lat_str=str(latitude))

    if "error" in predictions:
        return f"Erreur lors de l'obtention des prédictions: {predictions['error']}"
    
    # Filtrer les erreurs individuelles des modèles si nécessaire
    valid_predictions = {k: v for k, v in predictions.items() if "error" not in v}
    if not valid_predictions:
         error_messages = "; ".join([f"{k}: {v['error']}" for k, v in predictions.items() if "error" in v])
         return f"Toutes les prédictions ont échoué. Erreurs: {error_messages}"


    print(f"Prédictions obtenues: {valid_predictions}")
    
    # Formater les prédictions en une question pour le LLM
    user_query_from_predictions = format_predictions_for_query(valid_predictions)
    print(f"Requête construite pour le LLM: {user_query_from_predictions}")

    if user_query_from_predictions == "Aucune donnée de prédiction disponible.":
        return "Impossible de formuler une requête car aucune donnée de prédiction n'est disponible."

    # Obtenir la réponse du LLM
    return get_response(user_query_from_predictions)


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

if __name__ == "__main__":
    # Test de la nouvelle fonction
    # Remplacer par les coordonnées et la date souhaitées
    test_lon = "-100.0"  # Exemple de longitude
    test_lat = "40.0"    # Exemple de latitude
    # La date doit être dans le futur par rapport aux données d'entraînement J-1
    # et au format "YYYY-MM-DD"
    test_date = "2019-01-02" # Exemple de date future

    print(f"Test de la fonction get_agricultural_advice_with_predictions avec lon={test_lon}, lat={test_lat}, date={test_date}")
    advice = get_agricultural_advice_with_predictions(test_lon, test_lat, test_date)
    print("\n🤖 Conseil AgriConsult basé sur les prédictions:")
    print(advice)
