import pandas as pd
import re
import json
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from google import generativeai
import google.generativeai as genai
from google.generativeai import types
from fastapi import FastAPI, Body, Depends, Request
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="FlexAI", description="An AI-assistant for workspace booking")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NEARBY_PLACES_PROMPT = """
You are a helpful assistant that helps users find workspaces based on nearby places like cafes, restaurants, metro stations etc.
Your task is to analyze the user's query and identify only the workspace names from the dataset that are near the given place type within 500m radius.

Respond with a JSON array of strings like: [Space Name 1, Space Name 2, .....]

Eg - User queries: Show me day pass in delhi near a cafe
["Innov8 Connaught Place", "91springboard Nehru Place"]

IMPORTANT: Use only the data from the dataset provided.
IMPORTANT: Maintain contunuity with previous messages.

"""

async def parse_nearby_workspace(query: str, place_type: str, df_filtered: List[Dict], chat_history: List[Dict]) -> Dict[str, Any]:
    """Parse the workspaces about nearby places using Gemini."""
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Create the prompt
        prompt = f"""{NEARBY_PLACES_PROMPT}
        
        User query: {query}
        Place type: {place_type}
        Dataset: {df_filtered}
        Maintain continuity with previous messages
        Chat History: {chat_history}

        Respond with only the JSON array, no other text:"""
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        if response_text.startswith('```json'):
            response_text = response_text[response_text.find('['):response_text.rfind(']')+1]
        result = json.loads(response_text)
        print("[DEBUG] result: ", result)
        return result
    except Exception as e:
        print(f"Error parsing nearby query: {str(e)}")
        return []

@app.post("/api/nearbyplaces_chat")
async def nearbyplaces_chat(
    user_message: str = Body(..., embed=True),
    df_filtered: List[Dict] = Body(..., embed=True),
    chat_history: List[Dict] = Body([], embed=True),
    session_id: Optional[str] = Body(None, embed=True),
    user_id: Optional[str] = Body(None, embed=True),
    place_type: Optional[str] = Body(None, embed=True)
):
    """
    Endpoint to handle natural language queries about workspaces near specific places.
    Uses Gemini to understand the query and filter workspaces accordingly.
    """
    try:
        # Parse the user's query using Gemini
        filtered_results = await parse_nearby_workspace(user_message, place_type, df_filtered, chat_history)
        
        if not filtered_results:
            return {
                "filtered_results": df_filtered
            }

        matched_names = filtered_results  # now a list of names
        df = pd.DataFrame(df_filtered)
        df_filtered = df[df["Unboxed Coworking"].isin(matched_names)] # filter the dataframe based on matched names
            
        if len(filtered_results) == 0:
            response_text = " However, no workspaces match all your criteria. Would you like to try different filters?"
        
        return {
            "filtered_results": df_filtered
        }
        
    except Exception as e:
        print(f"Error in nearbyplaces_chat: {str(e)}")
        return { 
            "filtered_results": df_filtered,
            "error": str(e)
        }