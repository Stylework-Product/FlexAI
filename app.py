import os
import io
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import requests
#from htmlTemplates import css
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from typing import Optional, List
import google.generativeai as genai
import pickle

from connection_app import store_embeddings, load_embeddings

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

file_id = os.getenv("file_id")
EMBEDDING_NAME = "embeddings"  # Used for MongoDB storage

INITIAL_PROMPT = (
    "You are an expert assistant in helping user answer questions based on the document."
    "Answer the questions based on the provided document. But do not specify in the response 'based on the document' just answer like a normal assistant."
    "Also have a friendly conversation with user. All questions will not be related to the document."
    "Be concise and accurate."
)

class GeminiLLM(LLM):
    model: str = "models/gemini-1.5-flash"
    initial_prompt: str = INITIAL_PROMPT

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        full_prompt = f"{self.initial_prompt}\n\n{prompt}"
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(full_prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini-llm"

class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name: str = "models/embedding-001", api_key: str = None):
        self.model_name = model_name
        self.api_key = GEMINI_API_KEY
        genai.configure(api_key=self.api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model = self.model_name,
                content = text,
                task_type = "retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, texts: str) -> List[float]:
        result = genai.embed_content(
            model = self.model_name,
            content = texts,
            task_type = "retrieval_query"
        )
        return result["embedding"]

def get_text_chunks(text: str) -> List[str]:
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def fetch_pdf_bytes_from_gdrive(file_id) -> bytes:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download file: HTTP {response.status_code}")

def get_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    text = ""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_conversation_chain(vectorstore, initial_prompt=INITIAL_PROMPT):
    llm = GeminiLLM(initial_prompt=initial_prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("PDF not loaded yet.")
        return

    with st.chat_message("user"):
        st.markdown(user_question)

    response = st.session_state.conversation.invoke({"question": user_question})
    bot_msg = response["answer"]
    st.session_state.chat_history = response["chat_history"]

    with st.chat_message("assistant"):
        st.markdown(bot_msg)

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
    #st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDF Chatbot 🤖")

    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history):
            role = "user" if i % 2 == 0 else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

    user_question = st.chat_input("Ask a question about your document:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Document from Google Drive")
        if st.button("Load from DB"):
            with st.spinner("Loading embeddings from DB..."):
                faiss_bytes = load_embeddings(EMBEDDING_NAME)
                vectorstore = pickle.loads(faiss_bytes)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ Loaded from DB!")

        if st.button("Fetch & Store PDF"):
            with st.spinner("Downloading and embedding..."):
                pdf_bytes = fetch_pdf_bytes_from_gdrive(file_id)
                raw_text = get_pdf_text_from_bytes(pdf_bytes)
                chunks = get_text_chunks(raw_text)
                embeddings = GeminiEmbeddings()
                vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
                # Save FAISS index to MongoDB
                faiss_bytes = pickle.dumps(vectorstore)
                store_embeddings(EMBEDDING_NAME, faiss_bytes)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ PDF embedded and stored!")

if __name__ == "__main__":
    main()
