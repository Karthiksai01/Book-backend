from datetime import datetime
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

chat_collection = db.chat_history
voice_collection = db["voice_notes"]


def save_chat(user_id, document_id, question, answer):
    chat_collection.insert_one({
        "user_id": user_id,
        "document_id": document_id,
        "question": question,
        "answer": answer,
        "created_at": datetime.utcnow(),
        

    })
documents_collection = db.documents



def get_chat_history(user_id, document_id):
    chats = chat_collection.find(
        {"user_id": user_id, "document_id": document_id},
        {"_id": 0}
    )

    return [
        f"User: {c['question']}\nAI: {c['answer']}"
        for c in chats
    ]
def save_document(user_id: str, document_id: str, filename: str, text: str):
    documents_collection.insert_one({
        "user_id": user_id,
        "document_id": document_id,
        "filename": filename,
        "text": text,
        "created_at": datetime.utcnow()
    })


def get_document(document_id: str):
    return documents_collection.find_one(
        {"document_id": document_id},
        {"_id": 0}
    )

def get_user_documents(user_id: str):
    docs = documents_collection.find({"user_id": user_id}, {"_id": 0, "text": 0})
    return list(docs)


def get_document_history(user_id: str, document_id: str):
    chats = chat_collection.find(
        {"user_id": user_id, "document_id": document_id},
        {"_id": 0}
    ).sort("created_at", 1)

    return list(chats)

def save_voice_note(user_id: str, document_id: str, audio_url: str, script: str):
    voice_collection.update_one(
        {"user_id": user_id, "document_id": document_id},
        {"$set": {"audio_url": audio_url, "script": script}},
        upsert=True
    )


def get_voice_note(user_id: str, document_id: str):
    return voice_collection.find_one(
        {"user_id": user_id, "document_id": document_id},
        {"_id": 0}
    )


