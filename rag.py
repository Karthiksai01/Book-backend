import os
from pathlib import Path
from typing import List

#from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import os
from dotenv import load_dotenv
load_dotenv()
# ✅ local embedding model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


class LocalSentenceTransformerEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [ _st_model.encode(t).tolist() for t in texts ]

    def embed_query(self, text: str) -> List[float]:
        return _st_model.encode(text).tolist()


embeddings = LocalSentenceTransformerEmbeddings()

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vectorstores"
VECTOR_DIR.mkdir(exist_ok=True)


def build_and_save_vectorstore(document_id: str, document_text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_text(document_text)

    docs = [Document(page_content=c) for c in chunks]

    vectorstore = FAISS.from_documents(docs, embeddings)

    path = str(VECTOR_DIR / document_id)
    vectorstore.save_local(path)

    return vectorstore


def load_vectorstore(document_id: str):
    path = str(VECTOR_DIR / document_id)

    if not os.path.exists(path):
        return None

    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def retrieve_chunks(vectorstore, query: str, k: int = 3):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([r.page_content for r in results])
