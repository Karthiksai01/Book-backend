from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


# ==============================
# Local Embedding Model
# ==============================
_model = SentenceTransformer("all-MiniLM-L6-v2")


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return _model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return _model.encode([text])[0].tolist()


# Initialize embeddings
embeddings = LocalEmbeddings()


# ==============================
# Build Vectorstore (In Memory)
# ==============================
def build_vectorstore(document_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_text(document_text)
    docs = [Document(page_content=c) for c in chunks]

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore


# ==============================
# Retrieve Relevant Chunks
# ==============================
def retrieve_chunks(vectorstore, query: str, k: int = 3):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([r.page_content for r in results])
