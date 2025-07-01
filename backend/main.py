from fastapi import FastAPI, Body, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from connection import ChatSession, create_new_session, get_existing_session
from dotenv import load_dotenv
from google import generativeai
import google.generativeai as genai
from google.generativeai import types
from typing import Optional
from embeddings import GeminiEmbeddings
from embedding_manager import list_files_in_folder, EMBEDDING_NAME, folder_id
from fastapi.responses import StreamingResponse
from gemini_chat import gemini_chat
from general_chat import general_chat
from pdf_chat import pdf_chat
from utils import fetch_sheet_as_df, print_text_animated, get_session, global_sessions
 
app = FastAPI(title="FlexAI", description="An AI-assistant for workspace booking")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES"))
SHEET_API_KEY = os.getenv("SHEET_API_KEY")
SHEET_ID = os.getenv("SHEET_ID")  
SHEET_NAME = os.getenv("SHEET_NAME")  
EMBEDDING_NAME = "embeddings"

genai.configure(api_key=GEMINI_API_KEY)

@app.get("/session/{session_id}")
async def get_session_data(session_id: str):
    """Get session data by ID"""
    if session_id in global_sessions:
        return global_sessions[session_id].__dict__
    return {"error": "Session not found"}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in global_sessions:
        del global_sessions[session_id]
        return {"message": "Session deleted"}
    return {"error": "Session not found"}

class UserInfo(BaseModel):
    name: str
    email: str
    phone: str

class SessionRequest(BaseModel):
    user_info: UserInfo

@app.post("/session")
async def create_session(data: SessionRequest):
    user_info = data.user_info
    name = user_info.name.strip().lower().replace(" ", "_")
    email = user_info.email.strip().lower()
    user_id = f"{name}_{email}"
    
    # Create new session and store in global sessions
    session = create_new_session(user_id=user_id)
    global_sessions[session.session_id] = session
    
    print("[DEBUG] Created session:", session.session_id, "for user_id:", user_id)

    return {"session_id": session.session_id, "user_id": user_id}


if SHEET_API_KEY and SHEET_ID and SHEET_NAME:
    df = fetch_sheet_as_df(SHEET_ID, SHEET_NAME, SHEET_API_KEY)
else:
    print("Incorrect credentials - SHEET_API_KEY, SHEET_ID and SHEET_NAME")

# setup for router gemini - which routes the user query to 3 different gemini models
GEMINI_ROUTER_PROMPT = """
    You are a router assistant. Given a user message, decide if it is about:
    - "pdf" → if the question refers to anything about the company Stylework or functionalities (eg what is fixed memebership, what is flex ai, what are the features of flexboard etc.).
    - "workspace" → if it refers to workspace booking, area, city, location, budget, seats, coworking, offices, pricing etc. Any city name, area name, type of workspace should be classified as workspace.
    - "general" → if it's a general conversation or unrelated (greeting, jokes, questions about you, etc.)

    Only return one of the following exact outputs: "pdf", "workspace", or "general".
    Do not explain.

    User message: "{message}"
"""

def classify_intent_with_gemini(user_message: str) -> str:
    try:
        router_model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = GEMINI_ROUTER_PROMPT.format(message=user_message.strip())
        response = router_model.generate_content(prompt)
        intent = response.text.strip().lower()
        if intent in ["pdf", "workspace", "general"]:
            return intent
        else:
            return "general"
    except Exception as e:
        print("Router error:", e)
        return "general"

@app.post("/multimodal_agent")
async def multimodal_agent_router(
    user_message: str = Body(..., embed=True),
    chat_history: list = Body([], embed=True),
    session_id: Optional[str] = Body(None, embed=True),
    user_id: Optional[str] = Body(None, embed=True)
):
    intent = classify_intent_with_gemini(user_message)
    print(intent)
    if intent == "pdf":
        return await pdf_chat(user_message=user_message, chat_history=chat_history, session_id=session_id, user_id=user_id)
    elif intent == "workspace":
        return await gemini_chat(user_message=user_message, chat_history=chat_history, session_id=session_id, user_id=user_id)
    else:
        return await general_chat(user_message=user_message, chat_history=chat_history, session_id=session_id, user_id=user_id)
