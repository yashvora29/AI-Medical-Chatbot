from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

# Load the same embedding model used when creating ChromaDB
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Load existing ChromaDB
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)


# Load Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)


# User question
query = "What are the symptoms of diabetes?"


# Retrieve relevant chunks
results = vectorstore.similarity_search(query, k=3)


# Combine retrieved chunks
context = "\n\n".join(
    document.page_content
    for document in results
)


# Create prompt for Gemini
prompt = f"""
You are a medical information assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I could not find this information in the provided medical document."

Do not make up information.

Context:
{context}

Question:
{query}

Answer:
"""


# Generate answer using Gemini
response = llm.invoke(prompt)


print("\nQuestion:")
print(query)

print("\nAnswer:")
if isinstance(response.content, list):
    answer = response.content[0]["text"]
else:
    answer = response.content

print(answer)