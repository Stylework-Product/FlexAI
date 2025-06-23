import pandas as pd
import numpy as np
from fastapi import FastAPI, Body, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from fastapi.middleware.cors import CORSMiddleware
import re
import json
from typing import Optional, List
import os
import requests
from connection import ChatSession, create_new_session, get_existing_session
from datetime import datetime
from dotenv import load_dotenv
from google import generativeai
import google.generativeai as genai
from google.generativeai import types
import io
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from typing import Optional, List
import pickle
from embeddings import GeminiEmbeddings
from connection_app import load_embeddings
 
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
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE"))
SHEET_API_KEY = os.getenv("SHEET_API_KEY")
file_id = os.getenv("file_id")
EMBEDDING_NAME = "embeddings"

genai.configure(api_key=GEMINI_API_KEY)

INITIAL_PROMPT = (
    "You are an expert FlexAI assistant in helping user answer questions based on the document."
    "Stylework is India's largest flexible workspace provider, offering a robust solution for businesses of various sizes. With a presence in 100+ cities in India, we connect individuals, startups, and enterprises to a diverse network of ready-to-move-in coworking and managed office spaces."
    "Answer the questions based on the provided document. But do not mention that the response is 'based on the document', just answer like a normal assistant."
    "Also have a friendly conversation with user. All questions will not be related to the document."
    "Be concise and accurate."
)

class GeminiLLM(LLM):
    model: str = "models/gemini-2.0-flash"
    initial_prompt: str = INITIAL_PROMPT

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        full_prompt = f"{self.initial_prompt}\n\n{prompt}"
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(full_prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini-llm"

def get_text_chunks(text: str) -> List[str]:
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def get_conversation_chain(vectorstore, initial_prompt=INITIAL_PROMPT):
    llm = GeminiLLM(initial_prompt=initial_prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)

def handle_userinput(user_question, conversation, chat_history=None):
    if not user_question or not conversation:
        return "PDF not loaded yet."
    # Pass chat_history if provided
    inputs = {"question": user_question}
    if chat_history is not None:
        inputs["chat_history"] = chat_history
    response = conversation.invoke(inputs)
    bot_msg = response.get("answer", "")
    return bot_msg

class UserInput(BaseModel):
    workspaceName: Optional[str] = ''
    city: str
    area: Optional[List[str]] = []  
    workspaceType: str
    size: Optional[int] = 1
    amenities: Optional[List[str]] = []
    bundle: Optional[List[str]] = []    
    budget: Optional[int] = 0
    rating: Optional[int] = 0
    offeringType: Optional[str] = ''

class ChatMessage(BaseModel):
    id: str
    text: str
    sender: str
    timestamp: str

class ChatRequest(BaseModel):
    user_message: str
    chat_history: List[ChatMessage]
    session_id: Optional[str] = None

global_sessions = {}

def get_session(session_id: str = None, user_id: str = None):
    if session_id:
        if session_id in global_sessions:
            session = global_sessions[session_id]
        else:
            session = get_existing_session(session_id)
            if session:
                global_sessions[session_id] = session
            else:
                # Return None if session not found
                return None
    else:
        # Don't create sessions here - only retrieve existing ones
        return None

    return session

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

def fetch_sheet_as_df(sheet_id, sheet_name, api_key):
    url = (
        f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/{sheet_name}?key={api_key}"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    values = data.get("values", [])
    if not values:
        raise ValueError("No data found in the sheet.")
    df = pd.DataFrame(values[1:], columns=values[0])
    return df

SHEET_ID = os.getenv("SHEET_ID")  
SHEET_NAME = os.getenv("SHEET_NAME")  

if SHEET_API_KEY and SHEET_ID and SHEET_NAME:
    df = fetch_sheet_as_df(SHEET_ID, SHEET_NAME, SHEET_API_KEY)
else:
    print("Incorrect credentials - SHEET_API_KEY, SHEET_ID and SHEET_NAME")

numeric_columns = [
    'DAY PASS',
    'FLEXI DESK _MONTHLY',
    'DD_MONTHLY',
    'PC_MONTHLY',
    'review_count',
    'seats_available'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col])

df['avg_rating'] = pd.to_numeric(df['avg_rating']).astype(float)

def clean_amenity(a):
    return re.sub(r'[\[\]"\']', '', a).strip().lower()

df['Unboxed Coworking'] = df['Unboxed Coworking'].str.strip().str.lower()
df['CITY'] = df['CITY'].str.strip().str.lower()
df['STATUS'] = df['STATUS'].str.strip().str.lower()
df['CATEGORY AS PER PRICING'] = df['CATEGORY AS PER PRICING'].str.strip().str.lower()
df['CATEGORY AS PER SAP'] = df['CATEGORY AS PER SAP'].str.strip().str.lower()
if 'AREA' in df.columns:
    df['AREA'] = df['AREA'].str.strip().str.lower()

df['AMENITIES'] = df['AMENITIES'].apply(lambda x: [clean_amenity(s) for s in str(x).split(',')])

city_alias_map = {}
for city in df['CITY'].unique():
    city_alias_map[city] = city
    city_lower = city.strip().lower()
    # Only map well-known aliases, not substrings
    if city_lower == "delhi nct":
        city_alias_map["delhi"] = city
        city_alias_map["new delhi"] = city
    elif city_lower == "delhi":
        city_alias_map["new delhi"] = city
    if city_lower == "gurgaon":
        city_alias_map["gurugram"] = city
    if city_lower == "gurugram":
        city_alias_map["gurgaon"] = city
    if city_lower == "bengaluru":
        city_alias_map["bangalore"] = city
        city_alias_map["bangaluru"] = city
    if city_lower == "bangalore":
        city_alias_map["bengaluru"] = city
    if city_lower == "mumbai":
        city_alias_map["bombay"] = city
    if city_lower == "bombay":
        city_alias_map["mumbai"] = city
    if city_lower == "sahibzada ajit singh nagar (mohali)":
        city_alias_map["mohali"] = city
    if city_lower == "mysuru (mysore)":
        city_alias_map["mysore"] = city
        city_alias_map["mysuru"] = city
    if city_lower == "warangal (urban)":
        city_alias_map["warangal"] = city

GEMINI_ROUTER_PROMPT = """
    You are a router assistant. Given a user message, decide if it is about:
    - "pdf" → if the question refers to anything about the company or functionalities (eg what is fixed memebership, what is flex ai, what are the functionalities of ai etc.).
    - "workspace" → if it refers to workspace booking, location, budget, seats, coworking, offices, pricing etc.
    - "general" → if it's a general conversation or unrelated (greeting, jokes, questions about you, etc.)

    Only return one of the following exact outputs: "pdf", "workspace", or "general".
    Do not explain.

    User message: "{message}"
"""

def classify_intent_with_gemini(user_message: str) -> str:
    try:
        router_model = genai.GenerativeModel("gemini-2.0-flash")
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
def multimodal_agent_router(
    user_message: str = Body(..., embed=True),
    chat_history: list = Body([], embed=True),
    session_id: Optional[str] = Body(None, embed=True),
    user_id: Optional[str] = Body(None, embed=True)
):
    intent = classify_intent_with_gemini(user_message)
    print(intent)
    if intent == "pdf":
        return pdf_chat(user_message=user_message, chat_history=chat_history)
    elif intent == "workspace":
        return gemini_chat(user_message=user_message, chat_history=chat_history, session_id=session_id, user_id=user_id)
    else:
        return general_chat(user_message=user_message, chat_history=chat_history, session_id=session_id, user_id=user_id)

@app.post("/pdf_chat")
def pdf_chat(
    user_message: str = Body(..., embed=True),
    chat_history: list = Body([], embed=True),
    session_id: Optional[str] = Body(None, embed=True),
    user_id: Optional[str] = Body(None, embed=True)
):
    if not user_message:
        return {"reply": "Please provide a question."}
    faiss_bytes = load_embeddings(EMBEDDING_NAME)
    vectorstore = pickle.loads(faiss_bytes)
    conversation = get_conversation_chain(vectorstore)

    chat_sdk_history = []
    for msg in chat_history:
        sender = msg.get("sender")
        role = "user" if sender == "user" else "model"
        text = msg.get("text", "").strip()
        if text:
            chat_sdk_history.append({"role": role, "parts": [text]})

    reply_text = handle_userinput(user_message, conversation, chat_history=chat_history)

    if not reply_text or len(reply_text.strip()) < 3:
        reply_text = "Sorry, I couldn't process your request. You can contact the operation team regarding this query at operations@stylework.city!"

    session = get_session(session_id=session_id, user_id=user_id)
    if session is None and isinstance(user_id, str) and user_id:
        new_session = create_new_session(user_id=user_id)
        global_sessions[new_session.session_id] = new_session
        session = new_session
        session_id = session.session_id
    timestamp = None
    if session:
        session.add_user_message({
            "type": "user",
            "content": user_message
        })
        print(f"[DEBUG] User message added and saved to DB for session_id: {session_id}")
        session.add_assistant_message(reply_text, {}, None)
        print(f"[DEBUG] Assistant message added and saved to DB for session_id: {session_id}")
        last_msg = session.get_messages()[-1] if session.get_messages() else None
        timestamp = last_msg.get("timestamp") if last_msg else None
    else:
        timestamp = None
    
    return {"reply": reply_text, "session_id": session_id, "user_id": user_id, "timestamp": timestamp}
"""
def pdf_chat(
    user_message: str = Body(..., embed=True),
    chat_history: list = Body([], embed=True),
    session_id: Optional[str] = Body(None, embed=True),
    user_id: Optional[str] = Body(None, embed=True)
):
    if not user_message:
        return {"reply": "Please provide a question."}
    faiss_bytes = load_embeddings(EMBEDDING_NAME)
    vectorstore = pickle.loads(faiss_bytes)
    conversation = get_conversation_chain(vectorstore)
    
    response = handle_userinput(user_message, conversation, chat_history=chat_history)
    if not response or len(response.strip()) < 3:
        response = "Sorry, I couldn't process your request. You can contact the operation team regarding this query at operations@stylework.city!"

    session = get_session(session_id=session_id, user_id=user_id)
    if session is None and isinstance(user_id, str) and user_id:
        new_session = create_new_session(user_id=user_id)
        global_sessions[new_session.session_id] = new_session
        session = new_session
        session_id = session.session_id
    timestamp = None
    if session:
        session.add_user_message({
            "type": "user",
            "content": user_message
        })
        session.add_assistant_message(response, {}, None)
        last_msg = session.get_messages()[-1] if session.get_messages() else None
        timestamp = last_msg.get("timestamp") if last_msg else None
        
    return {"reply": response, "session_id": session_id, "user_id": user_id, "timestamp": timestamp}
"""

GENERAL_PROMPT = """
    You are FlexAI, a friendly assistant for Styleworks. Greet users, answer general questions, and guide them towards workspace booking or learning about Styleworks and Flexboard features.
    Stylework is India's largest flexible workspace provider, offering a robust solution for businesses of various sizes. With a presence in 100+ cities in India, we connect individuals, startups, and enterprises to a diverse network of ready-to-move-in coworking and managed office spaces.
    If the user asks about booking, features, or the platform, offer to help or provide information.
    Do not answer workspace-specific queries here; only handle general conversation.

    User message: {message}
    """

@app.post("/general_chat")
def general_chat(
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

    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = GENERAL_PROMPT.format(message=user_message.strip())

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

    return {"reply": reply_text, "timestamp": timestamp}


@app.post("/gemini_chat")
def gemini_chat(
    user_message: str = Body(..., embed=True),
    chat_history: list = Body([], embed=True),
    session_id: Optional[str] = Body(None, embed=True),
    user_id: Optional[str] = Body(None, embed=True)
):
    
    if len(chat_history) > MAX_HISTORY_MESSAGES:
        chat_history = chat_history[-MAX_HISTORY_MESSAGES:]

    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not set."}

    # Get existing session - don't create new ones here
    session = get_session(session_id=session_id, user_id=user_id)
    if session is None:
        return {"error": f"Session {session_id} not found. Please create a new session."}
    
    session_id = session.session_id
    print(f"Session ID: {session_id}")

    session.add_user_message({
        "type": "user",
        "content": user_message 
    })

    system_prompt = (
        "You are FlexiAI, a helpful assistant for a workspace booking platform called Styleworks. Your main job is to assist users in searching for and booking workspaces.\n"
        "Stylework is India's largest flexible workspace provider, offering a robust solution for businesses of various sizes. With a presence in 100+ cities in India, we connect individuals, startups, and enterprises to a diverse network of ready-to-move-in coworking and managed office spaces."
        "CRITICAL INSTRUCTIONS:\n"
        "1. DO NOT filter, search, or recommend any workspaces yourself\n"
        "2. DO NOT mention specific workspace names, addresses, or details\n"
        "3. DO NOT provide any workspace recommendations in your response\n"
        "4. Your ONLY job is to:\n"
        "	- Have a friendly conversation with the user\n"
        "	- Extract the following information and format as JSON: workspaceName, city, area, workspaceType (options: day pass, flexi desk, dedicated desk, private cabin), size, amenities (list), bundle (options: standard, silver, gold, platinum, platinum+), budget, rating, offeringType (options: day pass, flexi desk, dedicated desk, private cabin)\n"
        "	- Answer users' questions about the platform and the recommendations provided (eg if they ask about what amenities are provided by a specific workspace or the price of a workspace you should be able to answer it based on the information provided) and this is NOT a request for recommendation engine.\n"
        "	- You should also be able to answer basic questions like among the workspaces recommended which one has the best rating, which one is the cheapest, etc. This is NOT a workspace request so you do not have to search the recommendation engine.\n"
        "	- Extract search parameters from their message\n"
        "	- Let the recommendation engine handle ALL workspace suggestions\n"
        "	- Make sure user provides city and type of workspace they are looking for, if not provided in the start then ask them these questions one after another. After the initial requirements are provided ask them if they have a specific requirement in amentities, budget, area etc.\n"
        "	- If the workspace type is not specified in the start but is mentioned later, update the search parameters accordingly but if it is specified in the start and not explicitly told to change later then keep it same.\n"
        "	- If the user specifies a workspace type in the start, use that type for the search but if they mention it later then update the search parameters(workspaceType) accordingly.\n\n"
        "5. For **every workspace-related user message**, you MUST extract and return the full JSON object, even if only one field changes (e.g., just budget).\n"
        "6. DO NOT skip JSON during workspace-related user message just because it was already provided earlier. Always repeat the full updated JSON when any user parameter changes. ONLY for workspace requests\n"
        "7. You are a helpful assistant and reply to users in a friendly manner when the user doesnt ask workspace related queries. While having friendly conversation do not include JSON in your reply as it is not a workspace query.\n"
        "8. You DO NOT have to filter any workspace yourself, the engine has the capability to do so."
        "9. When user asks the price of a workspace based on the offering - the recommendation engine has the capability to do so. DO NOT think on your own."
        "When a user asks about workspaces:\n"
        "	- Acknowledge their request professionally\n"
        "	- If information is missing, set to null or empty values (0 for integer columns and [] for list columns)\n"
        "	- DO NOT include any specific workspace details or recommendations\n"
        "	- Extract the input parameters from the user message and format them as JSON.\n\n"
        "Understand important keywords:\n"
        "	- 'day pass' means a single day access\n"
        "	- 'flexi desk' means a flexible desk for a month\n"
        "	- 'dedicated desk' means a personal desk for a month\n"
        "	- 'private cabin' means a coworking space or workspace or office or private office or cabin or shared office for a month\n"
        "	- 'bundle' refers to the pricing category (standard, silver, gold, platinum, platinum+)\n"
        "	- 'budget' refers to the maximum price they are willing to pay\n"
        "	- 'offerings' refers to the type of desk types (day pass, flexi desk, dedicated desk, private cabin) provided by the workspace\n\n"
        "If user mentions any words like 'office', 'coworking space', 'shared office', 'workspace', 'desk', 'cabin', 'private office', etc., consider it as a workspace search request unless they ask about the services provided - understand if the request is a question or a statement and then decide accordingly.\n\n"
        "If the user asks about workspaces, extract their requirements and respond with a friendly message acknowledging their request.\n"
        "REMEMBER: "
        "	- Not all user messages will be about workspaces. If the user asks about something else, just have a friendly conversation with the knowledge provided to you. And do include JSON in your reply for this.\n\n"
        "	- The recommendation engine will add the actual workspace suggestions after your response.\n"
        "IMPORTANT: Maintain continuity with previous messages. If the user refers to something mentioned earlier, use that context in your response.\n\n"
        #f"IMPORTANT:  If user asks for filtering/sorting options (user_wants_filters={user_wants_filters}), mention that filtering options are available in your response and display the recommendations with the filtering options\n\n"
        f"User message: {user_message}\n"
        f"Chat history: {chat_history}\n"
        "-- JSON Enforcement Rule --"
        "When the user gives new input (like budget, location, size, etc.), REPLACE the previous value and OUTPUT the FULL updated JSON again."
        "Example - User: My budget is 450 rupees"
        "You must respond with:"
        """{
        "workspaceName": "",
        "city": "delhi",
        "area": [],
        "workspaceType": "day pass",
        "size": 1,
        "amenities": [],
        "bundle": ["gold"],
        "budget": 450,
        "rating": 0,
        "offeringType": "day pass"
        }"""
        "Always update the full structure. NEVER skip this JSON - for workspace request ONLY."
    )
    filtered_history = [msg for msg in chat_history if msg.get("sender") == "user"]

    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")

        # Build Gemini SDK-compatible chat history
        chat_sdk_history = [
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["Understood. I will follow the instructions."]}
        ]

        for msg in chat_history:
            sender = msg.get("sender")
            role = "user" if sender == "user" else "model"
            text = msg.get("text", "").strip()
            if text:
                chat_sdk_history.append({"role":role, "parts":[text]})

        # Start Gemini chat session
        gemini_chat_session = gemini_model.start_chat(history=chat_sdk_history)

        # Send message to Gemini
        gemini_response = gemini_chat_session.send_message(user_message)
        raw_reply = gemini_response.text

        response_metadata = {}
        tool_calls = None

        if not raw_reply or len(raw_reply.strip()) < 15:
            reply_text = "I understand your workspace requirements. Let me find the best options for you."
        else:
            reply_text = raw_reply

    except Exception as e:
        return {"error": f"Gemini SDK error: {str(e)}"}
        
    print(f"Gemini API response: {raw_reply}") 

    if not raw_reply:
        return {"error": "Empty response from Gemini API"}

    lines = raw_reply.split('\n')
    cleaned_lines = []
    workspace_keywords = [
        'workspaceName:', 'coworking space:', 'Address:', 'location:', 'bundle:', 'budget:', 'Workspace Type:',
        'Amenities:', 'Rating:', 'Seats available:', 'capacity:', 'size:', 'offeringType:',
        'visit:', 'book now:', 'contact:', 'city:', 'Offerings:', 'workspaceType:', 'Category:', 'Link:',
        'workspace name'
    ]
    
    for line in lines:
        line_lower = line.lower().strip()
        if any(keyword in line_lower for keyword in workspace_keywords):
            continue
        """
        if re.search(r'^\d+\.\s+[\w\s]+\(.*,.*\)', line.strip()):
            continue
        if re.search(r'\b\d+.*road|street|avenue|lane|sector\b', line_lower):
            continue
        """
        cleaned_lines.append(line)
    
    reply_text = '\n'.join(cleaned_lines).strip()

    if len(reply_text.strip()) < 15:
        reply_text = "I understand your workspace requirements. Let me find the best options for you."

    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_reply)
    extracted = None
    
    if json_match:
        reply_text = reply_text.replace(json_match.group(0), '').strip()
        reply_text = re.sub(r'```json|```', '', reply_text).strip()
        try:
            extracted = json.loads(json_match.group(0))
            if not isinstance(extracted, dict):
                extracted = None
        except json.JSONDecodeError:
            extracted = None

    final_reply = reply_text

    price_keywords = ["price", "cost", "rate", "charges"]
    offering_types = {
        "day pass": "DAY PASS",
        "flexi desk": "FLEXI DESK _MONTHLY",
        "dedicated desk": "DD_MONTHLY",
        "private cabin": "PC_MONTHLY"
    }

    if extracted and (extracted.get("offeringType") and extracted.get("workspaceName")):
        found_offering = str(extracted.get("offeringType") or "").strip().lower()
        workspace_name = str(extracted.get("workspaceName") or "").strip().lower()
        if found_offering not in offering_types:
            return {"reply": "I understand your workspace requirements, but I couldn't determine the workspace type. Please specify if you're looking for a day pass, flexi desk, dedicated desk, or private cabin."}
        
        df_temp = df.copy()
        res_row = df_temp[df_temp['Unboxed Coworking'].str.lower() == workspace_name]
        if res_row.empty:
            return {"reply": "I understand your workspace requirements, but I couldn't find the specified workspace. Please check the name and try again."}
        
        for key in res_row['Offering'].values[0].split(','):
            key = key.strip().lower()
            if key == found_offering:
                final_reply += f"\n\nI found the workspace '{workspace_name.title()}' with the offering type '{found_offering.title()}' has a price of ₹{res_row[offering_types[found_offering]].values[0]}."
                break

    if extracted and (extracted.get("city") or extracted.get("workspaceType") or extracted.get("area")):
        try:
            name = str(extracted.get("workspaceName") or "").strip().lower()
            raw_city = str(extracted.get("city") or "").strip().lower()
            city = city_alias_map.get(raw_city, raw_city)
            area_val = extracted.get("area")
            if isinstance(area_val, str):
                areas = [a.strip().lower() for a in re.split(r',|and', area_val) if a.strip()]
            elif isinstance(area_val, list):
                areas = [str(a).strip().lower() for a in area_val if str(a).strip()]
            else:
                areas = []

            workspace_type = str(extracted.get("workspaceType") or "").strip().lower()
            bundle = extracted.get("bundle")

            if isinstance(bundle, list):
                bundle = [str(b).strip().lower() for b in bundle if str(b).strip()]
            elif isinstance(bundle, str):
                bundle = [bundle.strip().lower()] if bundle.strip() else []
            else:
                bundle = []

            try:
                size = int(extracted.get("size") or 1)
                size = max(1, size)
            except (ValueError, TypeError):
                size = 1

            try:
                budget = float(extracted.get("budget") or 0)
                budget = max(0, budget)
            except (ValueError, TypeError):
                budget = 0
            try:
                rating = float(extracted.get("rating") or 0)
                rating = max(0, min(5, rating))  
            except (ValueError, TypeError):
                rating = 0

            user_amenities = extracted.get("amenities") or []
            if isinstance(user_amenities, list):
                user_amenities = [str(a).strip().lower() for a in user_amenities if str(a).strip()]
            else:
                user_amenities = []

            df_filtered = df.copy()

            if name and 'Unboxed Coworking' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['Unboxed Coworking'].str.lower() == name]

            if city and 'CITY' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['CITY'].str.lower() == city]

            if 'STATUS' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['STATUS'].str.lower() == 'live']

            if areas and 'AREA' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['AREA'].str.lower().isin(areas)]

            if 'seats_available' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['seats_available'] >= size]

            price_col_map = {
                'day pass': 'DAY PASS',
                'flexi desk': 'FLEXI DESK _MONTHLY',
                'dedicated desk': 'DD_MONTHLY',
                'private cabin': 'PC_MONTHLY'
            }
            price_col = price_col_map.get(workspace_type)

            if isinstance(bundle, str):
                bundle_list = [b.strip().lower() for b in bundle.split(',') if b.strip()]
            elif isinstance(bundle, list):
                bundle_list = [str(b).strip().lower() for b in bundle if str(b).strip()]
            else:
                bundle_list = []

            if bundle_list and 'CATEGORY AS PER PRICING' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['CATEGORY AS PER PRICING'].str.lower().isin(bundle_list)]

            if price_col and price_col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[price_col] > 0]
                if budget > 0:
                    df_filtered = df_filtered[df_filtered[price_col] <= budget]

            if rating > 0 and 'avg_rating' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['avg_rating'] >= rating]

            if df_filtered.empty:
                recommendations_text = "\n\nSorry, I couldn't find any workspaces matching your criteria. You might want to try adjusting your requirements."
                final_reply += recommendations_text
                return {"reply": final_reply}

            # --- Feature similarity calculations (always use all available features) ---
            similarities_list = []
            weights = []
            # 1. Amenities similarity (cosine)
            amenities_data = []
            if 'AMENITIES' in df_filtered.columns:
                for amenities in df_filtered['AMENITIES']:
                    if isinstance(amenities, list):
                        amenities_data.append([str(a).strip().lower() for a in amenities])
                    else:
                        amenities_data.append([])
                if amenities_data and user_amenities:
                    mlb = MultiLabelBinarizer()
                    services_encoded = mlb.fit_transform(amenities_data)
                    all_amenities = set(mlb.classes_)
                    user_amenities_filtered = [a for a in user_amenities if a in all_amenities]
                    if user_amenities_filtered:
                        user_cat = mlb.transform([user_amenities_filtered])
                        amenity_sim = cosine_similarity(user_cat, services_encoded)[0]
                        similarities_list.append(amenity_sim)
                        weights.append(1.0)
                    else:
                        similarities_list.append([0]*len(df_filtered))
                        weights.append(1.0)
            # 2. Area match (binary)
            if areas and 'AREA' in df_filtered.columns:
                area_sim = df_filtered['AREA'].apply(lambda x: 1.0 if x in areas else 0.0).values
                similarities_list.append(area_sim)
                weights.append(1.0)
            # 3. Bundle match (binary)
            if bundle_list and 'CATEGORY AS PER PRICING' in df_filtered.columns:
                bundle_sim = df_filtered['CATEGORY AS PER PRICING'].apply(lambda x: 1.0 if x in bundle_list else 0.0).values
                similarities_list.append(bundle_sim)
                weights.append(1.0)
            # 4. Rating (normalized difference)
            if rating > 0 and 'avg_rating' in df_filtered.columns:
                max_rating = 5.0
                rating_sim = 1.0 - abs(df_filtered['avg_rating'] - rating) / max_rating
                similarities_list.append(rating_sim.values)
                weights.append(1.0)
            # 5. Price (normalized difference)
            if price_col and price_col in df_filtered.columns and budget > 0:
                max_price = max(df_filtered[price_col].max(), budget)
                price_sim = 1.0 - abs(df_filtered[price_col] - budget) / max_price
                similarities_list.append(price_sim.values)
                weights.append(1.0)
            # 6. Workspace type (binary)
            if workspace_type and 'Offering' in df_filtered.columns:
                ws_type_sim = df_filtered['Offering'].apply(lambda x: 1.0 if workspace_type in str(x).lower() else 0.0).values
                similarities_list.append(ws_type_sim)
                weights.append(1.0)
            # --- Combine similarities ---
            if similarities_list:
                similarities_arr = np.array(similarities_list)
                weighted_sim = np.average(similarities_arr, axis=0, weights=weights)
                df_filtered = df_filtered.copy()
                df_filtered['similarity_score'] = (weighted_sim * 100).round(2)
                high_similarity = df_filtered[df_filtered['similarity_score'] >= MIN_SIMILARITY_SCORE]
                if not high_similarity.empty:
                    top_recommendations = high_similarity.sort_values(['similarity_score', 'avg_rating'], ascending=[False, False]).head(25)
                else:
                    top_recommendations = df_filtered.sort_values('avg_rating', ascending=False).head(25)
            else:
                top_recommendations = df_filtered.sort_values('avg_rating', ascending=False).head(25)

            seen_workspaces = set()
            result = []
            
            for _, row in top_recommendations.iterrows():
                workspace_name =  str(row.get('Unboxed Coworking', '')).strip()
                workspace_address = str(row.get('ADDRESS', '')).strip()
                workspace_id = f"{workspace_name.lower()}_{workspace_address.lower()}"
                
                if workspace_id not in seen_workspaces and workspace_name:
                    seen_workspaces.add(workspace_id)
                    
                    category_value = str(row.get('CATEGORY AS PER PRICING', '')).strip().title()
                    
                    price_value = ""
                    if price_col and price_col in row:
                        price_value = row.get(price_col, '')
                    
                    rec = {
                        "name": workspace_name,
                        "address": workspace_address,
                        "workspace_type": workspace_type or "general",
                        "city": str(row.get('CITY', '')).strip(),
                        "area": str(row.get('AREA', '')).strip(),
                        "amenities": row.get('AMENITIES', []) if isinstance(row.get('AMENITIES'), list) else [],
                        "status": str(row.get('STATUS', '')).strip(),
                        "seats_available": row.get('seats_available', ''),
                        "rating": row.get('avg_rating', ''),
                        "category": category_value,
                        "price": price_value,
                        "offerings": str(row.get('Offering', '')),
                        "similarity_score": row.get('similarity_score', 0)
                    }
                    result.append(rec)
            
            sort_by_price = False
            sort_price_pattern = re.compile(
                r'\b(?:sort|order|arrange|filter)\s+(?:the\s+)?(?:.*?)(?:price|cost|rate|charges)\b',
                re.IGNORECASE
            )
            if sort_price_pattern.search(user_message):
                sort_by_price = True

            if sort_by_price and result:
                def get_price(rec):
                    try:
                        return float(rec.get('price', float('inf')))
                    except (ValueError, TypeError):
                        return float('inf')
                result = sorted(result, key=get_price)

            sort_by_rating = False
            sort_rating_pattern = re.compile(
                r'\b(?:sort|order|arrange|filter)\s+(?:the\s+)?(?:.*?)(?:rating)\b',
                re.IGNORECASE
            )
            if sort_rating_pattern.search(user_message):
                sort_by_rating = True

            if sort_by_rating and result:
                def get_rating(rec):
                    try:
                        return float(rec.get('rating', float('-inf')))
                    except (ValueError, TypeError):
                        return float('inf')
                result = sorted(result, key=get_rating, reverse=True)
            
            if result:
                recommendations_text = "\n\nHere are some workspace recommendations for you:\n"
                for idx, rec in enumerate(result, 1):
                    recommendations_text += (
                        f"\n{idx}. {rec['name'].title()}"
                    )
                    if rec['area']:
                        recommendations_text += f" (Area: {rec['area'].title()})"
                    """
                    if rec['area'] and rec['city']:
                        recommendations_text += f" ({rec['area']}, {rec['city']})"
                    elif rec['city']:
                        recommendations_text += f" ({rec['city']})"
                    """
                    recommendations_text += f"\n   Address: {rec['address']}"

                    recommendations_text += f"\n   Workspace Type: {rec['workspace_type'].title()}"

                    recommendations_text += f"\n   Offerings: {rec['offerings']}"

                    if rec['amenities']:
                        amenities_str = ', '.join(rec['amenities'])  
                        recommendations_text += f"\n   Amenities: {amenities_str}"
                    
                    if rec['seats_available']:
                        recommendations_text += f"\n   Seats Available: {rec['seats_available']}"
                    
                    if rec['rating']:
                        recommendations_text += f"\n   Rating: {rec['rating']}"
                    
                    if rec['category']:
                        recommendations_text += f"\n   Category: {rec['category']}"

                    if rec['price']:
                        recommendations_text += f"\n   Price: ₹{rec['price']}"
                    
                    if rec['similarity_score'] > 70:
                        recommendations_text += f"\n   Similarity Score: {rec['similarity_score']}% "

                    workspace_slug = rec['name'].lower().replace(' ', '-').replace('&', 'and')
                    city_slug = rec['city'].lower().replace(' ', '-')
                    if rec['workspace_type'].lower() == "private cabin":
                        workspace_type_slug = "private-office-cabins"
                    else:
                        workspace_type_slug = rec['workspace_type'].replace(' ', '-')
                    
                    recommendations_text += f"\n   Link: [View Details](https://www.stylework.city/{workspace_type_slug}/{city_slug}/{workspace_slug})\n"
                
                final_reply += recommendations_text
            else:
                final_reply += "\n\nSorry, I couldn't find any workspaces matching your specific criteria. You might want to try adjusting your requirements."
        except Exception as e:
            print(f"Error in recommendation engine: {str(e)}")
            final_reply += "\n\nI understand your requirements, but encountered an issue while searching. Please try again."
    
    session.add_assistant_message(final_reply, response_metadata, tool_calls)
    # Get the last message (assistant) to extract its timestamp
    last_msg = session.get_messages()[-1] if session.get_messages() else None
    timestamp = last_msg.get("timestamp") if last_msg else None

    return {"reply": final_reply, "timestamp": timestamp}