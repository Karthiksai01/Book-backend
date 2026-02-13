from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import shutil
import uuid

from graph import app_graph
from models import (
    documents_collection,
    get_chat_history,
    save_document,
    get_document,
    get_user_documents,
    get_document_history
)

from utils import extract_text


app = FastAPI(title="StudyMate AI")

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://book-frontned-phi.vercel.app",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ===============================
# ROOT ROUTES
# ===============================

@app.get("/")
def home():
    return {"message": "StudyMate AI backend running ✅"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ===============================
# UPLOAD DOCUMENT
# ===============================

@app.post("/upload")
async def upload_document(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        file_bytes = await file.read()

        text = extract_text(file.filename, file_bytes)

        if not text.strip():
            return {"error": "Could not extract text from document."}

        document_id = str(uuid.uuid4())

        save_document(
            user_id=user_id,
            document_id=document_id,
            filename=file.filename,
            text=text
        )

        # ❌ Removed build_and_save_vectorstore
        # RAG is now built dynamically inside chat_agent

        return {
            "message": "Document uploaded successfully ✅",
            "document_id": document_id,
            "filename": file.filename
        }

    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}


# ===============================
# AGENT ROUTER
# ===============================

@app.post("/agent")
def run_agent(payload: dict):
    try:
        user_id = payload.get("user_id")
        document_id = payload.get("document_id")
        agent_type = payload.get("agent_type")

        if not user_id:
            return {"error": "user_id is required"}

        if not agent_type:
            return {"error": "agent_type is required"}

        valid_agents = ["chat", "summarize", "voice", "reference", "doubt"]

        if agent_type not in valid_agents:
            return {"error": f"Invalid agent_type. Use one of: {valid_agents}"}

        document_text = None

        if agent_type != "doubt":
            if not document_id:
                return {"error": "document_id is required"}

            doc = get_document(document_id)

            if not doc:
                return {"error": "Document not found. Please upload document first."}

            document_text = doc["text"]

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


# ===============================
# DOCUMENT ROUTES
# ===============================

@app.get("/documents/{user_id}")
def list_documents(user_id: str):
    try:
        return {"documents": get_user_documents(user_id)}
    except Exception as e:
        return {"error": f"Failed to fetch documents: {str(e)}"}


@app.get("/history/{user_id}/{document_id}")
def chat_history(user_id: str, document_id: str):
    try:
        return {"history": get_document_history(user_id, document_id)}
    except Exception as e:
        return {"error": f"Failed to fetch history: {str(e)}"}


@app.delete("/documents/{user_id}/{document_id}")
def delete_document(user_id: str, document_id: str):
    try:
        result = documents_collection.delete_one({
            "user_id": user_id,
            "document_id": document_id
        })

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")

        # ❌ Removed vectorstore folder deletion
        # Because we are no longer saving to disk

        return {"message": "Document deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# LOCAL RUN
# ===============================

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
