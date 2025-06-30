from langchain.embeddings.base import Embeddings
import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class GeminiEmbeddings(Embeddings):
    """Wrapper around Gemini Embeddings API for document and query embeddings."""
    def __init__(self, model_name: str = "models/embedding-001", api_key: str = None):
        self.model_name = model_name
        self.api_key = GEMINI_API_KEY
        genai.configure(api_key=self.api_key)

    # embed documents using Gemini API
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings

    # embed user queries using Gemini API
    def embed_query(self, texts):
        result = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type="retrieval_query"
        )
        return result["embedding"]