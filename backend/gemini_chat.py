import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import re
import json
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from google import generativeai
import google.generativeai as genai
from google.generativeai import types
from fastapi import APIRouter, Body, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from utils import fetch_sheet_as_df, print_text_animated, get_session
from embedding_manager import EMBEDDING_NAME, folder_id
from nearbyplaces_chat import nearbyplaces_chat
from connection import ChatSession, create_new_session, get_existing_session

router = APIRouter()

load_dotenv()

# format for user input
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
    placeType: Optional[str] = ''

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES"))
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE"))
SHEET_API_KEY = os.getenv("SHEET_API_KEY")
SHEET_ID = os.getenv("SHEET_ID")  
SHEET_NAME = os.getenv("SHEET_NAME")  
EMBEDDING_NAME = "embeddings"
NO_OF_SPACES = os.getenv("NO_OF_SPACES")

genai.configure(api_key=GEMINI_API_KEY)

if SHEET_API_KEY and SHEET_ID and SHEET_NAME:
    df = fetch_sheet_as_df(SHEET_ID, SHEET_NAME, SHEET_API_KEY)
else:
    print("Incorrect credentials - SHEET_API_KEY, SHEET_ID and SHEET_NAME")

# data processing of dataset
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

WORKSPACE_TYPE_SLUG = {
    "day pass": "day-pass",
    "meeting room": "meeting-rooms",
    "private cabin": "private-office-cabin",
    "dedicated desk": "dedicated-desk",
    "open desk": "open-desk",
    "virtual office": "virtual-office"
}

VALID_CITIES = {
    "agartala", "agra", "ahmedabad", "ajmer", "akola", "ambala", "amritsar", "anand", "ankleshwar", 
    "balasinor", "bareilly", "bengaluru", "bhagalpur", "bhopal", "bhubaneshwar", "chakpachuria", 
    "chandigarh", "chennai", "chittoor", "coimbatore", "deesa", "dehradun", "delhi nct", "dhanbad", 
    "dharamshala", "dhule", "dimapur", "dubai", "durg", "east godavari", "east khasi hills", 
    "ernakulam", "faridabad", "gautam buddha nagar", "ghaziabad", "goa", "gorakhpur", "guntur", 
    "gurgaon", "guwahati", "gwalior", "hyderabad", "imphal west", "indore", "jabalpur", "jaipur", 
    "jalandhar", "jammu", "jamshedpur", "jodhpur", "kakkanad", "kanpur nagar", "kochi", "kolkata", 
    "kothaguda", "kottayam", "kozhikode", "kurnool", "lucknow", "ludhiana", "madurai", "mangalore", 
    "mohali", "mumbai", "mysuru (mysore)", "nagpur", "nandurbar", "nashik", "navi mumbai", "noida", 
    "palakkad", "panaji", "panchkula", "patna", "pondicherry", "prayagraj", "pune", "raipur", 
    "rajkot", "ranchi", "ratlam", "sahibzada ajit singh nagar (mohali)", "salem", "sangli", "sikar", 
    "siliguri", "surat", "thane", "thiruvananthapuram", "tiruchirappalli", "udaipur", "ujjain", 
    "vadodara", "varanasi", "vellore", "vijayawada", "visakhapatnam", "warangal (urban)", "zirakpur"
}

CATEGORY_IDS = {
    "standard": "63c8ef67b593488ed624bff4",
    "silver": "63c8ef6eb593488ed624bff5",
    "gold": "63c8ef74b593488ed624bff6",
    "platinum": "63c8ef7ab593488ed624bff7",
    "platinum+": "659c22a8c5737f2fe35d0d37"
}

SORT_BY_PRICE = {
    "low to high": "Price%20(Low%20to%20High)",
    "high to low": "Price%20(High%20to%20Low)"
}

def get_day_pass_budget(budget):
    """Map the given budget to the nearest lower day pass price point"""
    if budget < 200:
        return None
    elif 200 <= budget < 400:
        return 200
    elif 400 <= budget < 600:
        return 400
    elif 600 <= budget < 800:
        return 600
    elif 800 <= budget < 1000:
        return 800
    elif 1000 <= budget < 1200:
        return 1000
    elif 1200 <= budget < 1400:
        return 1200
    elif 1400 <= budget < 1600:
        return 1400
    elif 1600 <= budget < 1800:
        return 1600
    elif 1800 <= budget <= 2000:
        return 1800
    else:
        return 1800 

def get_space_budget(budget):
    """Map the given budget to the nearest lower space budget price point"""
    if budget < 3000:
        return None
    elif 3000 <= budget < 6000:
        return 3000
    elif 6000 <= budget < 9000:
        return 6000
    elif 9000 <= budget < 12000:
        return 9000
    elif 12000 <= budget < 15000:
        return 12000
    elif 15000 <= budget < 18000:
        return 15000
    elif 18000 <= budget <= 20000:
        return 18000
    else:
        return 18000

def generate_stylework_url(city: str, workspace_type: str, bundle: List[str], sort_by_price: bool, budget: int) -> str:
    """Generate Stylework.city URL based on extracted information"""
    city_slug = city.lower().replace(' ', '-')
    if workspace_type == "private cabin":
        type_slug = "private-office-cabins"
    else:
        type_slug = workspace_type.lower().replace(' ', '-')

    if not type_slug or not city_slug:
        return ""
    
    base_url = f"https://www.stylework.city/{type_slug}/{city_slug}"
    
    params = []
    
    if bundle:
        for category in bundle:
            category_id = CATEGORY_IDS.get(category)
            if category_id:
                params.append(f"category={category_id}")
    if budget:
        if workspace_type == "day pass":
            budget_range = get_day_pass_budget(budget)
            if budget_range:
                params.append(f"budget={budget_range}")
        else:
            space_range = get_space_budget(budget)
            if space_range:
                params.append(f"budget={space_range}")
    # Add sort preference
    if sort_by_price:
        sort_param = SORT_BY_PRICE.get("low to high")
        if sort_param:
            params.append(f"sortBy={sort_param}")
    
    # Combine URL with parameters
    if params:
        base_url += "?" + "&".join(params)
    
    return base_url

def format_workspace_recommendations(result: List[Dict[str, Any]]) -> str:
    if not result:
        return "\n\nSorry, I couldn't find any workspaces matching your specific criteria. You might want to try adjusting your requirements."

    recommendations_text = "\n\nHere are some workspace recommendations for you:\n"

    for idx, rec in enumerate(result, 1):
        recommendations_text += f"\n{idx}. {rec['name'].title()}"

        if rec.get('area'):
            recommendations_text += f" (Area: {rec['area'].title()})"

        recommendations_text += f"\n   Address: {rec.get('address', 'Not available')}"

        recommendations_text += f"\n   Workspace Type: {rec.get('workspace_type', '').title()}"

        offerings = rec.get('offerings')
        if offerings:
            recommendations_text += f"\n   Offerings: {offerings}"

        amenities = rec.get('amenities')
        if amenities:
            amenities_str = ', '.join(amenities)
            recommendations_text += f"\n   Amenities: {amenities_str}"

        if rec.get('seats_available'):
            recommendations_text += f"\n   Seats Available: {rec['seats_available']}"

        if rec.get('rating'):
            recommendations_text += f"\n   Rating: {rec['rating']}"

        if rec.get('category'):
            recommendations_text += f"\n   Category: {rec['category']}"

        if rec.get('price'):
            recommendations_text += f"\n   Price: ₹{rec['price']}"

        if rec.get('similarity_score', 0) > 70:
            recommendations_text += f"\n   Similarity Score: {rec['similarity_score']}%"

        # Constructing the dynamic workspace link
        name_slug = rec['name'].lower().replace(' ', '-').replace('&', 'and')
        city_slug = rec.get('city', '').lower().replace(' ', '-')
        workspace_type = rec.get('workspace_type', '').lower()

        if workspace_type == "private cabin":
            type_slug = "private-office-cabins"
        else:
            type_slug = workspace_type.replace(' ', '-')

        link = f"https://www.stylework.city/{type_slug}/{city_slug}/{name_slug}"
        recommendations_text += f"\n   Link: [View Details]({link})\n"

    return recommendations_text

# setup for the recommendation gemini model
@router.post("/gemini_chat")
async def gemini_chat(
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
        "You are FlexAI, a helpful assistant for a workspace booking platform called Styleworks. Your main job is to assist users in searching for and booking workspaces.\n"
        "Stylework is India's largest flexible workspace provider, offering a robust solution for businesses of various sizes. With a presence in 100+ cities in India, we connect individuals, startups, and enterprises to a diverse network of ready-to-move-in coworking and managed office spaces."
        "CRITICAL INSTRUCTIONS:\n"
        "1. DO NOT filter, search, or recommend any workspaces yourself\n"
        "2. DO NOT mention specific workspace names, addresses, or details\n"
        "3. DO NOT provide any workspace recommendations in your response\n"
        "4. Your ONLY job is to:\n"
        "	- Have a friendly conversation with the user like basic greetings, conversation etc. While having friendly conversation do not include JSON in your reply as it is not a workspace query.\n"
        "	- Extract the following information and format as JSON: workspaceName, city, area, workspaceType (options: day pass, flexi desk, dedicated desk, private cabin), size, amenities (list), bundle (also called category) (list - options: standard, silver, gold, platinum, platinum+), budget, rating, offeringType (options: day pass, flexi desk, dedicated desk, private cabin), placeType (cafe, resturant, bank etc.)\n"
        "	- Answer users' questions about the platform and the recommendations provided (eg if they ask about what amenities are provided by a specific workspace or the price of a workspace you should be able to answer it based on the information provided) and this is NOT a request for recommendation engine.\n"
        "	- Extract search parameters from their message\n"
        "	- Let the recommendation engine handle ALL workspace suggestions\n"
        "	- Make sure user provides city and type of workspace they are looking for, if not provided in the start then ask them these questions one after another. After the initial requirements are provided ask them if they have a specific requirement in amentities, budget, area etc.\n"
        "	- If the workspace type is not specified in the start but is mentioned later, update the search parameters accordingly but if it is specified in the start and not explicitly told to change later then keep it same.\n"
        "	- If the user specifies a workspace type in the start, use that type for the search but if they mention it later then update the search parameters(workspaceType) accordingly.\n\n"
        "5. DO NOT skip JSON during workspace-related user message just because it was already provided earlier. Always repeat the full updated JSON when any user parameter changes. ONLY for workspace requests\n"
        "6. When user asks the price of a workspace based on the offering - the recommendation engine has the capability to do so. DO NOT think on your own."
        "7. When a user asks about workspaces:\n"
        "	- Acknowledge their request professionally\n"
        "	- If information is missing, set to null or empty values (0 for integer columns and [] for list columns)\n"
        "	- DO NOT include any specific workspace details or recommendations\n"
        "	- Extract the input parameters from the user message and format them as JSON.\n\n"
        "8.Understand important keywords:\n"
        "	- 'day pass' means a single day access\n"
        "	- 'flexi desk' means a flexible desk for a month\n"
        "	- 'dedicated desk' means a personal desk for a month\n"
        "	- 'private cabin' means a coworking space or workspace or office or private office or cabin or shared office for a month\n"
        "	- 'bundle' refers to the pricing category (standard, silver, gold, platinum, platinum+)\n"
        "	- 'budget' refers to the maximum price they are willing to pay\n"
        "	- 'offerings' refers to the type of desk types (day pass, flexi desk, dedicated desk, private cabin) provided by the workspace\n\n"
        "   - 'placeType' refers to the type of place user wants the workspace to be near by (cafe, restaurant, bank etc)\n"
        "9.If user mentions any words like 'office', 'coworking space', 'shared office', 'workspace', 'desk', 'cabin', 'private office', etc., consider it as a workspace search request unless they ask about the services provided - understand if the request is a question or a statement and then decide accordingly.\n\n"
        "10.If the user wants to BOOK a workspace - ask the user for details (such as name, email, number of seats required, joining date of space etc or any other appropriate details needed) - only after the user had requested to look for a workspace, if workspace request was not iniitalized then go for the request. After collecting the details mention that the details are sent to the operations team and they will contact the user soon regarding the workspace query."
        "11.Make sure the responses are displayed in a neat format with proper spacing and formatting:"
        "   - Use bullet points (•) for lists"
        "   - Use **bold** for headings and important terms"
        "   - Add proper line breaks between sections"
        "   - Remove unnecessary spaces and formatting issues"
        "IMPORTANT: Any user query which is not in the 11 points functionality of the chatbot display the bot message - You can contact the operation team regarding this query at operations@stylework.city!"
        "IMPORTANT: Maintain continuity with previous messages. If the user refers to something mentioned earlier, use that context in your response.\n\n"
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
        "offeringType": "day pass",
        "placeType": "metro station"
        }"""
        "Always update the full structure. NEVER skip this JSON - for workspace request ONLY."
    )
    filtered_history = [msg for msg in chat_history if msg.get("sender") == "user"]

    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")

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
        'workspaceName:', 'coworking space:', 'Address:', 'location:', 'bundle:', 'budget:',
        'Amenities:', 'Rating:', 'Seats available:', 'capacity:', 'size:', 'offeringType:',
        'visit:', 'book now:', 'contact:', 'Offerings:', 'workspaceType:', 'Category:', 'Link:',
        'workspace name'
    ]
    
    # remove recommendation (if any) from the gemini response
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

    # extract the json structure from the gemini response
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

    # if user wants to know price of a particular workspace (eg what is day pass rate of Purple Coworking)
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
    
    # the main recommendation engine setup - starts by extracting input from the user query
    if extracted and (extracted.get("city") and extracted.get("workspaceType")):
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

            # perform basic filtering based on the inputs
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
                return print_text_animated(final_reply)

            # Check for location-based query (e.g., "show me day pass in delhi near a cafe")
            location_based_query = False
            place_type = None
            
            # First check if we have placeType in the extracted JSON
            if extracted and 'placeType' in extracted and extracted['placeType']:
                place_type = extracted['placeType']
                location_based_query = True
            
            if location_based_query and place_type and city:
                print(f"Workspace type: {workspace_type}, City: {city}, Place type: {place_type}")
                response = await nearbyplaces_chat(user_message, df_filtered, chat_history, session_id, user_id, place_type)
                df_filtered = response["filtered_results"]
                #print(df_filtered.head(10).to_string(max_colwidth=200, max_rows=None, max_cols=None))

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
            # --- Combine similarities with weights ---
            if similarities_list:
                similarities_arr = np.array(similarities_list)
                weighted_sim = np.average(similarities_arr, axis=0, weights=weights)
                df_filtered = df_filtered.copy()
                df_filtered['similarity_score'] = (weighted_sim * 100).round(2)
                high_similarity = df_filtered[df_filtered['similarity_score'] >= MIN_SIMILARITY_SCORE]
                if not high_similarity.empty:
                    top_recommendations = high_similarity.sort_values(['similarity_score', 'avg_rating'], ascending=[False, False]).head(int(NO_OF_SPACES))
                else:
                    top_recommendations = df_filtered.sort_values('avg_rating', ascending=False).head(int(NO_OF_SPACES))
            else:
                top_recommendations = df_filtered.sort_values('avg_rating', ascending=False).head(int(NO_OF_SPACES))

            seen_workspaces = set()
            result = []
            
            # storing the recommendations in a variable
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
            
            # sort by price functionality
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

            # sort by rating functionality 
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
            
            final_response = format_workspace_recommendations(result)
            stylework_url = generate_stylework_url(city, workspace_type, bundle, sort_by_price, budget)
            if stylework_url:
                final_response += f"\n🔗 **Browse more options:** {stylework_url}\n"
                print(f"[DEBUG] Generated Stylework URL: {stylework_url}")
            final_reply += final_response


        except Exception as e:
            print(f"Error in recommendation engine: {str(e)}")
            final_reply += "\n\nI understand your requirements, but encountered an issue while searching. Please try again."
    
    session.add_assistant_message(final_reply, response_metadata, tool_calls)
    # Get the last message (assistant) to extract its timestamp
    last_msg = session.get_messages()[-1] if session.get_messages() else None
    timestamp = last_msg.get("timestamp") if last_msg else None

    return {"reply": final_reply, "timestamp": timestamp}