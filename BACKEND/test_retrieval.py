from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

# Load the local embedding model (HuggingFace) and the Chroma vector store
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,  
    embedding_function=embeddings
)

#ask a question
query = "What are the symptoms of diabetes?"

results = vectorstore.similarity_search(query, k=3)

print("\nQuestion:", query)
print("\nRelevant information:\n")
for i, document in enumerate(results):
    print(f"--- Result {i + 1} ---")
    print(document.page_content[:1000])
    print()