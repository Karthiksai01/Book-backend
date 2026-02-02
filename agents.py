from llm import llm
from models import save_chat, save_voice_note, get_voice_note
from rag import build_vectorstore, retrieve_chunks

import os
import uuid
from gtts import gTTS

from search import web_search, youtube_search


# In-memory vectorstore cache (Render-safe)
VECTORSTORE_CACHE = {}


# ==============================
# 📚 Chat Agent (RAG)
# ==============================
def chat_agent(state):
    document_text = state.get("document_text") or ""
    user_query = state.get("user_query") or ""

    if not document_text.strip():
        state["output"] = "No document text provided. Please upload a document."
        return state

    if not user_query.strip():
        state["output"] = "Please ask a question."
        return state

    doc_id = state["document_id"]

    # Use in-memory cache
    vectorstore = VECTORSTORE_CACHE.get(doc_id)

    if vectorstore is None:
        vectorstore = build_vectorstore(document_text)
        VECTORSTORE_CACHE[doc_id] = vectorstore

    retrieved_context = retrieve_chunks(vectorstore, user_query)

    prompt = f"""
You are StudyBook AI (student assistant).
Answer ONLY using the context below.
If the answer is not present, say: Not found in document.

Context:
{retrieved_context}

Question:
{user_query}
"""

    response = llm.invoke(prompt)

    save_chat(
        state["user_id"],
        state["document_id"],
        user_query,
        response.content
    )

    state["output"] = response.content
    return state


# ==============================
# 📝 Summarizer Agent
# ==============================
def summarizer_agent(state):
    doc_text = state.get("document_text") or ""

    if not doc_text.strip():
        state["output"] = "No document found to summarize."
        return state

    prompt = f"""
You are StudyMate AI for students.

Task:
Summarize the document in a clean study format.

Output Format:
1) 5-Line Summary
2) Key Points (6 to 10 bullets)
3) Important Terms (5 to 8 terms with 1-line meaning)
4) Quick Revision (3 to 5 lines)

Rules:
- Use simple language
- Don't add extra sections
- Don't use special characters like # or **
- Keep it clean and readable

Document:
{doc_text}
"""

    response = llm.invoke(prompt)
    state["output"] = response.content
    return state


# ==============================
# 🔊 Voice Agent
# ==============================
def voice_agent(state):
    user_id = state.get("user_id")
    doc_id = state.get("document_id")
    doc_text = state.get("document_text") or ""

    # Check cache in MongoDB
    cached = get_voice_note(user_id, doc_id)
    if cached:
        state["output"] = {
            "audio_url": cached["audio_url"],
            "script": cached["script"],
            "cached": True
        }
        return state

    if not doc_text.strip():
        state["output"] = "No document found to generate voice explanation."
        return state

    prompt = f"""
You are StudyBook AI.

Create a voice-note style explanation.

Duration:
3 to 4 minutes (450–650 words)

Rules:
- Explain like teaching a student
- Use simple language
- No headings
- No bullet points
- Natural narration style

Document:
{doc_text}
"""

    response = llm.invoke(prompt)
    script_text = response.content.strip()

    audio_id = str(uuid.uuid4())
    filename = f"{audio_id}.mp3"

    audio_folder = "static/audio"
    os.makedirs(audio_folder, exist_ok=True)

    audio_path = os.path.join(audio_folder, filename)

    tts = gTTS(text=script_text, lang="en")
    tts.save(audio_path)

    audio_url = f"/static/audio/{filename}"

    save_voice_note(user_id, doc_id, audio_url, script_text)

    state["output"] = {
        "audio_url": audio_url,
        "script": script_text,
        "cached": False
    }

    return state


# ==============================
# 🌐 Reference Agent
# ==============================
def reference_agent(state):
    topic = state.get("user_query") or ""

    if not topic.strip():
        state["output"] = {
            "websites": [],
            "youtube": []
        }
        return state

    websites = web_search(topic, max_results=5)
    youtube = youtube_search(topic, max_results=4)

    state["output"] = {
        "websites": websites,
        "youtube": youtube
    }

    return state



# ==============================
# ❓ Doubt Agent
# ==============================
def doubt_agent(state):
    query = state.get("user_query") or ""
    history = state.get("chat_history") or []

    prompt = f"""
You are StudyMate AI - Doubt Clarifier.

Explain in simple language.
Give step-by-step clarification.
Use short examples if needed.
Be friendly and clear.

Conversation History:
{history}

Student Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    state["output"] = response.content
    return state
