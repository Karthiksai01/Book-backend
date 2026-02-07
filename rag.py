import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# In-memory vector store (Render safe)
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


def retrieve_chunks(vectorstore, query: str, k: int = 3):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([r.page_content for r in results])
