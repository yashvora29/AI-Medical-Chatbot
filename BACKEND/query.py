from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "chroma_db"

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

question = input("Ask a medical question: ")

results = vectorstore.similarity_search(
    question,
    k=3
)

print("\nRelevant information:\n")

for i, result in enumerate(results):
    print(f"--- Result {i + 1} ---")
    print(result.page_content)
    print()