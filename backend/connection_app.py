import os
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import google.generativeai as genai
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")  
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
#MONGO_DB = os.getenv("MONGO_DB")
MONGO_DB = "pdf_chatbot"

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
    print(f"Loading embeddings from GridFS with name: {name}")
    file = fs.find_one({"filename": name})
    print(f"File found: {file is not None}")
    if not file:
        raise ValueError(f"No file named '{name}' found in GridFS.")
    return file.read()
