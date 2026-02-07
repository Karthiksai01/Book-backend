import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ==============================
# Load Environment
# ==============================
load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ==============================
# Gemini Custom Embedding Class
# ==============================
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return response["embedding"]


# Initialize embedding model
embeddings = GeminiEmbeddings()


# ==============================
# In-Memory Vector Store (Render Safe)
# ==============================
def build_vectorstore(document_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_text(document_text)
    docs = [Document(page_content=c) for c in chunks]

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

    return vectorstore


# ==============================
# Retrieval
# ==============================
def retrieve_chunks(vectorstore, query: str, k: int = 3):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([r.page_content for r in results])
