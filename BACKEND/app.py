from pathlib import Path
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

load_dotenv(PROJECT_DIR / ".env")

app = Flask(
    __name__,
    template_folder=str(PROJECT_DIR / "FRONTEND")
)

CHROMA_PATH = BASE_DIR / "chroma_db"

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load ChromaDB
vectorstore = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings
)

# Load Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()
    query = data.get("question")

    if not query:
        return jsonify({
            "error": "Question is required"
        }), 400

    # Retrieve relevant chunks
    results = vectorstore.similarity_search(
        query,
        k=2
    )

    # Combine retrieved context
    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # Prompt Gemini
    prompt = f"""
You are a medical information assistant.
Answer the question using ONLY the provided context.
Do not make up information.
If the answer cannot be found in the context, say:
"I could not find this information in the provided medical document."
The document is provided for informational purposes and
should not be treated as a substitute for professional medical advice.
Context:
{context}
Question:
{query}
Answer:
"""

    response = llm.invoke(prompt)

    # Extract Gemini text
    if isinstance(response.content, list):
        answer = response.content[0]["text"]
    else:
        answer = response.content
    return jsonify({
        "question": query,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(debug=True)