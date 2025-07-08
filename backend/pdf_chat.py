from typing import Optional, List, Dict, Any
import os
import requests
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from connection_app import load_embeddings
import pickle
from google import generativeai
import google.generativeai as genai
from google.generativeai import types
from fastapi import APIRouter, Body, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from utils import fetch_sheet_as_df, get_session, print_text_animated
from connection import ChatSession, create_new_session, get_existing_session

router = APIRouter()

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_NAME = "embeddings"

genai.configure(api_key=GEMINI_API_KEY)

# Gemini set up for pdf document answering
INITIAL_PROMPT = (
    "You are an expert FlexAI assistant in helping user answer questions based on the document."
    "Stylework is India's largest flexible workspace provider, offering a robust solution for businesses of various sizes. With a presence in 100+ cities in India, we connect individuals, startups, and enterprises to a diverse network of ready-to-move-in coworking and managed office spaces."
    "Answer the questions based on the provided document. But do not mention that the response is 'based on the document', just answer like a normal assistant."
    "Also have a friendly conversation with user. All questions will not be related to the document."
    "If the user query asked is not available within your domain knowledge then response should be - You can contact the operation team regarding this query at operations@stylework.city!"
    "REQUIRED: Make sure the responses are displayed in a neat format with proper spacing and formatting:"
    "- Use bullet points (•) for lists"
    "- Use **bold** for headings and important terms"
    "- Add proper line breaks between sections"
    "- Remove unnecessary spaces and formatting issues"
    "Be concise and accurate."
    "IMPORTANT: Maintain continuity with previous messages"
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

# splits the document into chunks
def get_text_chunks(text: str) -> List[str]:
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# initializes a llm gemini model and creates a memory for it
def get_conversation_chain(vectorstore, initial_prompt=INITIAL_PROMPT):
    llm = GeminiLLM(initial_prompt=initial_prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)

# recives the user query and produces a response
def handle_userinput(user_question, conversation, chat_history):
    if not user_question or not conversation:
        return "PDF not loaded yet."
    history = [(msg["role"], msg["parts"][0]) for msg in chat_history if msg.get("parts")]
    response = conversation.invoke({
        "question": user_question,
        "chat_history": history
    })
    bot_msg = response.get("answer", "You can contact the operation team regarding this query at operations@stylework.city!")
    return bot_msg

@router.post("/pdf_chat")
async def pdf_chat(
    user_message: str = Body(..., embed=True),
    chat_history: list = Body([], embed=True),
    session_id: Optional[str] = Body(None, embed=True),
    user_id: Optional[str] = Body(None, embed=True),
    global_sessions: Dict[str, ChatSession] = Depends(get_existing_session)
):
    if not user_message:
        return {"reply": "Please provide a question."}
    
    # loading embeddings for a single file
    faiss_bytes = load_embeddings(EMBEDDING_NAME)
    vectorstore = pickle.loads(faiss_bytes)
    conversation = get_conversation_chain(vectorstore)

    # loading embeddings for multiple files in a folder
    """
    file_list = list_files_in_folder(folder_id)
    merged_vectorstore = None
    for _file_id, file_name in file_list:
        embedding_key = f"{EMBEDDING_NAME}_{file_name}"
        try:
            faiss_bytes = load_embeddings(embedding_key)
            vectorstore = pickle.loads(faiss_bytes)
            if merged_vectorstore is None:
                merged_vectorstore = vectorstore
            else:
                merged_vectorstore.merge_from(vectorstore)
        except Exception as e:
            print(f"[WARN] Could not load embedding for {file_name}: {e}")
    if merged_vectorstore is None:
        return {"reply": "No embeddings found for any file."}
    conversation = get_conversation_chain(merged_vectorstore)
    """
    # storing response in chat history
    chat_sdk_history = []
    for msg in chat_history:
        sender = msg.get("sender")
        role = "user" if sender == "user" else "model"
        text = msg.get("text", "").strip()
        if text:
            chat_sdk_history.append({"role": role, "parts": [text]})

    session = get_session(session_id=session_id, user_id=user_id)
    if session is None and isinstance(user_id, str) and user_id:
        session = create_new_session(user_id=user_id)
        global_sessions[session.session_id] = session
        session_id = session.session_id
    
    if session:
        session.add_user_message({
            "type": "user",
            "content": user_message
        })
    
    session_messages = session.get_messages() if session else chat_history
    # retrieve the response
    reply_text = handle_userinput(user_message, conversation, chat_history=session_messages)

    if session:
        session.add_assistant_message(reply_text, {}, None)
        last_msg = session.get_messages()[-1] if session.get_messages() else None
        timestamp = last_msg.get("timestamp") if last_msg else None
    else:  
        timestamp = None
    if not reply_text or len(reply_text.strip()) < 3:
        reply_text = "You can contact the operation team regarding this query at operations@stylework.city!"
    print(f"[DEBUG] Gemini Response: {reply_text}")
    return print_text_animated(reply_text)
    #return {"reply": reply_text, "session_id": session_id, "user_id": user_id, "timestamp": timestamp, "chat_history": session.get_messages() if session else []}