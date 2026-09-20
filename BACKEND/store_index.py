from pathlib import Path
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "medical_book.pdf"
CHROMA_PATH = BASE_DIR / "chroma_db"

#here we define a function to load the PDF and extract its text content. The function uses the PdfReader class from the pypdf 
# library to read the PDF file and extract text from each page. The extracted text is concatenated into
#  a single string, which is then returned.    
def load_pdf():
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

text = load_pdf()

def split_text(text , chunk_size = 1000 , overlap = 200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap 
    return chunks

#load pdf
text = load_pdf()

print("PDF loaded successfully!")
print("Characters extracted:", len(text))

#split pdf into chunks 
chunks = split_text(text)

print("Total chunks:", len(chunks))

#create local embedding model (huggingface)
embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)
print("Embedding model loaded!")

#converted chunks into langchain docs
documents = [
    Document(page_content=chunk)
    for chunk in chunks
]

#stored them in chromadb
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)

print("ChromaDB created successfully!")