import requests
import pandas as pd
from fastapi.responses import StreamingResponse
import asyncio

# fetch the dataset from google sheet
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

async def animate(text: str):
    for char in text:
        yield char
        await asyncio.sleep(0.005)  # adjust speed here

def print_text_animated(text: str):
    return StreamingResponse(animate(text), media_type="text/plain")

global_sessions = {}

def get_session(session_id: str = None, user_id: str = None):
    from connection import get_existing_session  # avoid circular import
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
