import os
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import google.generativeai as genai

load_dotenv()

MONGO_URI = "mongodb+srv://intern:p8IHZyhgpAvqRCBX@chatbotdb.mrfwk3h.mongodb.net/?retryWrites=true&w=majority&appName=ChatbotDB"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
#MONGO_DB = os.getenv("MONGO_DB")
MONGO_DB = "chatbotdb"

# Log the values for debugging
print(f"[DEBUG] MONGO_URI: {MONGO_URI}")
print(f"[DEBUG] MONGO_DB: {MONGO_DB}")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
fs = gridfs.GridFS(db)

def store_embeddings(name: str, data: bytes):
    existing = fs.find({"filename": name})
    for file in existing:
        fs.delete(file._id)
    fs.put(data, filename=name)

def load_embeddings(name: str) -> bytes:
    file = fs.find_one({"filename": name})
    print(f"Database name: {db.name}")
    print(f"Files in GridFS: {[file.filename for file in fs.find()]}")
    print(f"Loading embeddings from GridFS with name: {name}")
    file = fs.find_one({"filename": name})
    print(f"File found: {file is not None}")
    if not file:
        raise ValueError(f"No file named '{name}' found in GridFS.")
    return file.read()