from langchain_ollama import OllamaLLM as Ollama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Load embeddings and vectorstore
embeddings = OllamaEmbeddings(model="mistral")
vectorstore = FAISS.load_local("vectorstore/", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = Ollama(
    model="mistral",
    temperature=0.7,
    top_p=0.9,
    top_k=50
)


def get_response(user_query):
    print("--- Début de la fonction get_response ---")
    print(f"Requête utilisateur reçue : '{user_query}'")

    if not user_query.strip():
        print("La requête utilisateur est vide ou ne contient que des espaces.")
        print("--- Fin de la fonction get_response (requête vide) ---")
        return "Please enter a valid question."

    print("Détection de la langue...")
    try:
        language = detect(user_query)
        print(f"Langue détectée : {language}")
    except LangDetectException:
        language = "en"  # default to English if detection fails
        print("Échec de la détection de la langue, utilisation de l'anglais par défaut.")

    print("Récupération des documents pertinents (contexte)...")
    context = retriever.invoke(user_query)
    context_text = "\n".join([doc.page_content for doc in context])
    print(f"Contexte récupéré : \n---\n{context_text}\n---")

    print("Création des instructions pour le persona...")
    if language == "fr":
        persona_instructions = """
You are "Droplets", an AI expert in water scarcity. You speak clearly, concisely, and kindly—like a friendly environmental scientist. Stay focused only on water-related topics: water access, droughts, agriculture, climate impacts, clean water, sanitation, and resource management.

Politely decline off-topic questions and encourage rephrasing.

Behavior Guide:

Greet warmly but briefly.

Thank kindly if the user thanks you.

If facts come from documents, mention it humbly.

Use documents and history to inform responses.

Never make up facts. If unsure, say so clearly.

Maximum response: 60 words.
"""
        print("Instructions du persona en français sélectionnées.")
    else:
        persona_instructions = """
YYou are "Droplets", an AI expert in water scarcity. You speak clearly, concisely, and kindly—like a friendly environmental scientist. Stay focused only on water-related topics: water access, droughts, agriculture, climate impacts, clean water, sanitation, and resource management.

Politely decline off-topic questions and encourage rephrasing.

Behavior Guide:

Greet warmly but briefly.

Thank kindly if the user thanks you.

If facts come from documents, mention it humbly.

Use documents and history to inform responses.

Never make up facts. If unsure, say so clearly.

Maximum response: 60 words.
"""
        print("Instructions du persona en anglais sélectionnées.")

    print("Construction du prompt final...")
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
    print(f"Prompt final construit : \n---\n{prompt}\n---")

    print("Invocation du modèle LLM...")
    # Note : Vous appelez llm.invoke deux fois. Pour l'efficacité et la cohérence,
    # il est préférable de l'appeler une seule fois et de stocker le résultat.
    response_text = llm.invoke(prompt).strip()
    print(f"Réponse brute du LLM : '{response_text}'")
    
    print("--- Fin de la fonction get_response ---")
    return response_text

# Basic CLI
def chat():
    print("\n🌊 Droplets Chatbot - Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower().strip() == "exit":
                print("👋 Goodbye! Stay curious about the water Scarcity.")
                break
            response = get_response(user_input)
            print("Bot:", response)
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Stay curious about the water scarcity.")
            break
