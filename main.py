from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from models import documents_collection
import os, shutil
import json
import time

import uuid

from graph import app_graph
from models import (
    get_chat_history,
    save_document,
    get_document,
    get_user_documents,
    get_document_history
)

from utils import extract_text
from rag import build_and_save_vectorstore

app = FastAPI(title="StudyMate AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",   
        "http://127.0.0.1:5173"    
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")




# ✅ Home Route (avoids 404 on "/")
@app.get("/")
def home():
    return {"message": "StudyMate AI backend running ✅"}


# ✅ Health Route
@app.get("/health")
def health():
    return {"status": "ok"}


# ✅ Upload Document Route (PDF/DOCX/TXT)
@app.post("/upload")
async def upload_document(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        file_bytes = await file.read()

        # ✅ Extract text
        text = extract_text(file.filename, file_bytes)

        if not text.strip():
            return {"error": "Could not extract text from document."}

        # ✅ Create unique document_id
        document_id = str(uuid.uuid4())

        # ✅ Save to MongoDB
        save_document(
            user_id=user_id,
            document_id=document_id,
            filename=file.filename,
            text=text
        )

        # ✅ Build & save vectorstore immediately (RAG)
        build_and_save_vectorstore(document_id, text)

        return {
            "message": "Document uploaded successfully ✅",
            "document_id": document_id,
            "filename": file.filename
        }

    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}


# ✅ Main Agent Router Endpoint
@app.post("/agent")
def run_agent(payload: dict):
    try:
        # ✅ Extract fields
        user_id = payload.get("user_id")
        document_id = payload.get("document_id")
        agent_type = payload.get("agent_type")

        # ✅ Validations
        if not user_id:
            return {"error": "user_id is required"}

        if not agent_type:
            return {"error": "agent_type is required"}

        # ✅ Allowed agents
        valid_agents = ["chat", "summarize", "voice", "reference", "doubt"]

        if agent_type not in valid_agents:
            return {"error": f"Invalid agent_type. Use one of: {valid_agents}"}

        # ✅ For doubt agent → document is OPTIONAL
        document_text = None
        if agent_type != "doubt":
            if not document_id:
                return {"error": "document_id is required"}

            # ✅ Fetch document from DB
            doc = get_document(document_id)
            if not doc:
                return {"error": "Document not found. Please upload document first."}

            document_text = doc["text"]

        # ✅ Prepare state for LangGraph
        state = {
            "user_id": user_id,
            "document_id": document_id or "no_doc",
            "agent_type": agent_type,
            "user_query": payload.get("user_query"),
            "document_text": document_text,
            "chat_history": get_chat_history(user_id, document_id) if document_id else [],
            "output": None,
        }

        result = app_graph.invoke(state)

        return {"result": result.get("output")}

    except Exception as e:
        return {"error": f"Agent execution failed: {str(e)}"}

# ✅ List All Documents Uploaded by User
@app.get("/documents/{user_id}")
def list_documents(user_id: str):
    try:
        return {"documents": get_user_documents(user_id)}
    except Exception as e:
        return {"error": f"Failed to fetch documents: {str(e)}"}


# ✅ Get Chat History for One Document
@app.get("/history/{user_id}/{document_id}")
def chat_history(user_id: str, document_id: str):
    try:
        return {"history": get_document_history(user_id, document_id)}
    except Exception as e:
        return {"error": f"Failed to fetch history: {str(e)}"}


@app.delete("/documents/{user_id}/{document_id}")
def delete_document(user_id: str, document_id: str):
    try:
        print("DELETE REQUEST:", user_id, document_id)

        result = documents_collection.delete_one({
            "user_id": user_id,
            "document_id": document_id
        })

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")

        vector_path = f"vectorstores/{document_id}"
        if os.path.exists(vector_path):
            shutil.rmtree(vector_path)

        return {"message": "Document deleted successfully"}

    except Exception as e:
        print("❌ DELETE ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)