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

if __name__ == "__main__":
    chat()