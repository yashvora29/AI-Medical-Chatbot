from pathlib import Path

from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "medical_book.pdf"
CHROMA_PATH = BASE_DIR / "chroma_db"


def load_pdf():
    reader = PdfReader(PDF_PATH)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# Load PDF
text = load_pdf()

print("PDF loaded successfully!")
print("Characters extracted:", len(text))


# Split text
chunks = split_text(text)

print("Total chunks:", len(chunks))


# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# Convert chunks into LangChain documents
documents = [
    Document(page_content=chunk)
    for chunk in chunks
]


# Create ChromaDB
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=str(CHROMA_PATH)
)

print("ChromaDB created successfully!")