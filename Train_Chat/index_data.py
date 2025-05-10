from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# Load your JSON dataset with correct jq_schema according to the provided JSON file structure
loader = JSONLoader(
    file_path='./dataset/merged_dataset.json',
    jq_schema='.[].content',  # Extract only the 'content' field for embeddings
    text_content=True
)

# Load documents from JSON
documents = loader.load()

# Initialize Mistral embeddings via Ollama
embeddings = OllamaEmbeddings(model="mistral")

# Create FAISS vector store from documents
vectorstore = FAISS.from_documents(documents, embeddings)

# Save FAISS vectorstore locally
vectorstore.save_local("vectorstore/")

print("✅ Dataset indexed successfully!")
