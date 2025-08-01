from supabase import create_client, Client
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
  
class ChatSession:
    def __init__(self, user_id=None):
        self.session_id = str(uuid4())
        self.user_id = user_id
        self.message = []
        # Only save to database if user_id is provided (not empty sessions)
        if user_id:
            print(f"[DEBUG] Database connected. Session created with session_id: {self.session_id}")
            self.save_session()
        else:
            print(f"[DEBUG] Skipping session creation for empty user_id. Generated session_id: {self.session_id}")

    def save_session(self):
        # Only save if user_id exists and is not empty
        if not self.user_id:
            print(f"[DEBUG] Skipping database save for empty session: {self.session_id}")
            return
            
        session_data = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message": json.dumps(self.message)
        }

        existing = supabase.table("n8n_chat_histories").select("session_id").eq("session_id", self.session_id).execute()
        if existing.data:
            supabase.table("n8n_chat_histories").update(session_data).eq("session_id", self.session_id).execute()
            print(f"[DEBUG] Updated session in DB: {self.session_id}")
        else:
            supabase.table("n8n_chat_histories").insert(session_data).execute()
            print(f"[DEBUG] Inserted new session in DB: {self.session_id}")

    def add_user_message(self, content):
        self.message.append({
            "type": "user",
            "content": content,
            "additional_kwargs": {},
            "response_metadata": {},
            "timestamp": datetime.now().isoformat()
        })
        self.save_session()
        # Debug: User message sent to DB
        print(f"[DEBUG] User message added and saved to DB for session_id: {self.session_id}")
 
    def add_assistant_message(self, content, response_metadata, tool_calls, invalid_tool_calls=None):
        self.message.append({
            "type": "assistant",
            "content": content,
            "tool_calls": tool_calls or [],
            "additional_kwargs": {},
            "response_metadata": response_metadata,
            "invalid_tool_calls": invalid_tool_calls or [],
            "timestamp": datetime.now().isoformat()
        })
        self.save_session()
        # Debug: Assistant message sent to DB
        print(f"[DEBUG] Assistant message added and saved to DB for session_id: {self.session_id}")

    @staticmethod
    def get_session(session_id):
        result = supabase.table("n8n_chat_histories").select("*").eq("session_id", session_id).execute()
        if result.data:
            return ChatSession.from_dict(result.data[0])
        return None

    @staticmethod
    def from_dict(data):
        session = ChatSession(user_id=data.get("user_id"))
        session.session_id = data["session_id"]
        session.message = json.loads(data["message"])
        return session

    def get_messages(self):
        return self.message

def create_new_session(user_id=None):
    print(f"[DEBUG] create_new_session called with user_id: {user_id}")
    return ChatSession(user_id=user_id)

def get_existing_session(session_id):
    return ChatSession.get_session(session_id)