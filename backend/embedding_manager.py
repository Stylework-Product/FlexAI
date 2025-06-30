import os
import io
import pickle
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from googleapiclient.discovery import build
from google.oauth2 import service_account
from embeddings import GeminiEmbeddings  # Make sure this class is in a separate file
from connection_app import store_embeddings, load_embeddings  # Your Mongo store/load functions

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
file_id = os.getenv("file_id")
EMBEDDING_NAME = "embeddings"  # Key name used in MongoDB
folder_id = os.getenv("folder_id")
GOOGLE_DRIVE_API = os.getenv("GOOGLE_DRIVE_API")

def get_drive_service():
    """Authenticate and return the Google Drive service object."""
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_file(GOOGLE_DRIVE_API, scopes=scopes)
    return build('drive', 'v3', credentials=creds)

def list_files_in_folder(folder_id: str) -> list:
    """List all file IDs in a Google Drive folder."""
    service = get_drive_service()
    query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    return [(f['id'], f['name']) for f in files]

def fetch_pdf_bytes_from_gdrive(file_id: str) -> bytes:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download PDF from Google Drive. Status code: {response.status_code}")

def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def split_text_to_chunks(text: str, chunk_size=1000, chunk_overlap=200) -> List[str]:
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

"""
def build_and_store_embeddings_for_folder(folder_id: str, embedding_name_prefix: str = EMBEDDING_NAME):
    files = list_files_in_folder(folder_id)
    print(f"Found {len(files)} files in folder.")
    for file_id, file_name in files:
        print(f"\nProcessing file: {file_name} ({file_id})")
        try:
            pdf_bytes = fetch_pdf_bytes_from_gdrive(file_id)
            text = extract_pdf_text(pdf_bytes)
            chunks = split_text_to_chunks(text)
            embedder = GeminiEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embedding=embedder)
            faiss_bytes = pickle.dumps(vectorstore)
            store_embeddings(f"{embedding_name_prefix}_{file_name}", faiss_bytes)
            print(f"✅ Embeddings stored for {file_name}.")
        except Exception as e:
            print(f"❌ Failed to process {file_name}: {e}")

def load_all_vectorstores_from_db(folder_id: str, embedding_name_prefix: str = EMBEDDING_NAME):
    files = list_files_in_folder(folder_id)
    vectorstores = {}
    for _file_id, file_name in files:
        embedding_name = f"{embedding_name_prefix}_{file_name}"
        try:
            print(f"Loading FAISS index for {file_name} from MongoDB...")
            faiss_bytes = load_embeddings(embedding_name)
            from embeddings import GeminiEmbeddings  # Must import before unpickling
            vectorstore = pickle.loads(faiss_bytes)
            vectorstores[file_name] = vectorstore
            print(f"✅ Loaded FAISS index for {file_name}.")
        except Exception as e:
            print(f"❌ Failed to load embeddings for {file_name}: {e}")
    return vectorstores
"""
#build_and_store_embeddings_for_folder(folder_id, EMBEDDING_NAME)
#load_all_vectorstores_from_db(folder_id,EMBEDDING_NAME)

def build_and_store_embeddings(file_id: str, embedding_name: str = EMBEDDING_NAME):
    print("Fetching PDF...")
    pdf_bytes = fetch_pdf_bytes_from_gdrive(file_id)
    
    print("Extracting text...")
    text = extract_pdf_text(pdf_bytes)
    
    print("Splitting into chunks...")
    chunks = split_text_to_chunks(text)
    
    print("Generating embeddings...")
    embedder = GeminiEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embedder)

    print("Serializing FAISS index...")
    faiss_bytes = pickle.dumps(vectorstore)

    print("Storing to MongoDB...")
    store_embeddings(embedding_name, faiss_bytes)

    print("✅ Embeddings stored successfully.")

def load_vectorstore_from_db(embedding_name: str = EMBEDDING_NAME) -> FAISS:
    print("Loading FAISS index from MongoDB...")
    faiss_bytes = load_embeddings(embedding_name)
    from embeddings import GeminiEmbeddings  # Must import before unpickling
    vectorstore = pickle.loads(faiss_bytes)
    print("✅ FAISS index loaded successfully.")
    return vectorstore

#build_and_store_embeddings(file_id, EMBEDDING_NAME)
load_vectorstore_from_db(EMBEDDING_NAME)
