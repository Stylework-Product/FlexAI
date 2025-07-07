from fastapi import APIRouter, Body, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import os
import requests
from dotenv import load_dotenv
from google import generativeai
import google.generativeai as genai
from google.generativeai import types
from embeddings import GeminiEmbeddings
from connection_app import load_embeddings
from utils import print_text_animated, get_session
from connection import ChatSession, create_new_session, get_existing_session

router = APIRouter()

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES"))

genai.configure(api_key=GEMINI_API_KEY)

# setup for the general gemini model
GENERAL_PROMPT = """
    You are FlexAI, a friendly assistant for Styleworks. Greet users, answer general questions.
    If the user asks about booking, features, or the platform, offer to help or provide information.
    Do not answer workspace-specific queries here; only handle general conversation.
    
    IMPORTANT: Format your responses properly:
    - Use **bold** for headings and important terms
    - Add proper line breaks between sections
    - Remove unnecessary spaces and formatting issues

    User message: {message}
    """

@router.post("/general_chat")
async def general_chat(
    user_message: str = Body(..., embed=True),
    chat_history: list = Body([], embed=True),
    session_id: Optional[str] = Body(None, embed=True),
    user_id: Optional[str] = Body(None, embed=True)
):
    if len(chat_history) > MAX_HISTORY_MESSAGES:
        chat_history = chat_history[-MAX_HISTORY_MESSAGES:]

    session = get_session(session_id=session_id, user_id=user_id)
    if session is None and session_id:
        return {"error": f"Session {session_id} not found. Please create a new session."}

    # initialize gemini model for general chat
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = GENERAL_PROMPT.format(message=user_message.strip())

    # prepare and store messages in chat history
    chat_sdk_history = []
    for msg in chat_history:
        sender = msg.get("sender")
        role = "user" if sender == "user" else "model"
        text = msg.get("text", "").strip()
        if text:
            chat_sdk_history.append({"role": role, "parts": [text]})

    gemini_chat_session = model.start_chat(history=chat_sdk_history)
    gemini_response = gemini_chat_session.send_message(prompt)
    reply_text = gemini_response.text.strip()

    if session:
        session.add_user_message({
            "type": "user",
            "content": user_message
        })
        session.add_assistant_message(reply_text, {}, None)

        last_msg = session.get_messages()[-1] if session.get_messages() else None
        timestamp = last_msg.get("timestamp") if last_msg else None
    else:
        timestamp = None

    return print_text_animated(reply_text)
    #return {"reply": reply_text, "timestamp": timestamp}
