import os
import re
import json
import asyncio
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from connection import create_new_session, get_existing_session
import urllib.parse

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES"))
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE"))

genai.configure(api_key=GEMINI_API_KEY)

# URL generation mappings
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

def normalize_city_name(city: str) -> str:
    """Normalize city name for URL generation"""
    if not city:
        return ""
    
    city_lower = city.lower().strip()
    
    # Handle special cases
    if city_lower in ["delhi", "new delhi"]:
        return "delhi-nct"
    elif city_lower in ["mysore", "mysuru"]:
        return "mysuru-mysore"
    elif city_lower in ["mohali"]:
        return "sahibzada-ajit-singh-nagar-mohali"
    
    # Replace spaces with hyphens and handle special characters
    normalized = city_lower.replace(" ", "-").replace("(", "").replace(")", "")
    
    # Check if normalized city exists in our valid cities list
    if normalized in [c.replace(" ", "-").replace("(", "").replace(")", "") for c in VALID_CITIES]:
        return normalized
    
    return city_lower.replace(" ", "-")

def extract_workspace_info_from_recommendations(recommendations_text: str) -> Dict[str, Any]:
    """Extract workspace information from recommendations text for URL generation"""
    info = {
        "workspace_types": set(),
        "cities": set(),
        "categories": set(),
        "sort_preference": None
    }
    
    if not recommendations_text:
        return info
    
    lines = recommendations_text.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        
        # Extract workspace types
        for workspace_type in WORKSPACE_TYPE_SLUG.keys():
            if workspace_type in line_lower:
                info["workspace_types"].add(workspace_type)
        
        # Extract cities
        for city in VALID_CITIES:
            if city in line_lower:
                info["cities"].add(city)
        
        # Extract categories
        for category in CATEGORY_IDS.keys():
            if category in line_lower:
                info["categories"].add(category)
        
        # Extract sort preferences
        if "price" in line_lower:
            if "low to high" in line_lower or "ascending" in line_lower:
                info["sort_preference"] = "low to high"
            elif "high to low" in line_lower or "descending" in line_lower:
                info["sort_preference"] = "high to low"
    
    return info

def generate_stylework_url(workspace_info: Dict[str, Any]) -> str:
    """Generate Stylework.city URL based on extracted information"""
    if not workspace_info["workspace_types"] or not workspace_info["cities"]:
        return ""
    
    # Use the first workspace type and city found
    workspace_type = list(workspace_info["workspace_types"])[0]
    city = list(workspace_info["cities"])[0]
    
    # Get the slug for workspace type
    workspace_slug = WORKSPACE_TYPE_SLUG.get(workspace_type, "day-pass")
    
    # Normalize city name
    city_slug = normalize_city_name(city)
    
    # Build base URL
    base_url = f"https://www.stylework.city/{workspace_slug}/{city_slug}"
    
    # Add query parameters
    params = []
    
    # Add categories
    if workspace_info["categories"]:
        for category in workspace_info["categories"]:
            category_id = CATEGORY_IDS.get(category)
            if category_id:
                params.append(f"category={category_id}")
    
    # Add sort preference
    if workspace_info["sort_preference"]:
        sort_param = SORT_BY_PRICE.get(workspace_info["sort_preference"])
        if sort_param:
            params.append(f"sortBy={sort_param}")
    
    # Combine URL with parameters
    if params:
        base_url += "?" + "&".join(params)
    
    return base_url

async def animate(text: str):
    for char in text:
        yield char
        await asyncio.sleep(0.005)  # adjust speed here

def print_text_animated(text: str):
    return StreamingResponse(animate(text), media_type="text/plain")

# Gemini set up for pdf document answering
INITIAL_PROMPT = (
    "You are an expert assistant in helping user answer questions based on the document."
    "Answer the questions based on the provided document. But do not specify in the response 'based on the document' just answer like a normal assistant."
    "Also have a friendly conversation with user. All questions will not be related to the document."
    "Be concise and accurate."
    "IMPORTANT: Maintain continuity with previous messages"
)

class GeminiLLM:
    def __init__(self, model: str = "models/gemini-2.0-flash-lite", initial_prompt: str = INITIAL_PROMPT):
        self.model = model
        self.initial_prompt = initial_prompt

    def _call(self, prompt: str) -> str:
        full_prompt = f"{self.initial_prompt}\n\n{prompt}"
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(full_prompt)
        return response.text

# Router prompt for intent classification
GEMINI_ROUTER_PROMPT = """
    You are a router assistant. Given a user message, decide if it is about:
    - "pdf" → if the question refers to anything about the company or functionalities (eg what is fixed memebership, what is flex ai, what are the functionalities of ai etc.).
    - "workspace" → if it refers to workspace booking, area, city, location, budget, seats, coworking, offices, pricing etc.
    - "general" → if it's a general conversation or unrelated (greeting, jokes, questions about you, etc.)

    Respond with only one word: "pdf", "workspace", or "general"

    User message: {message}
"""

def classify_intent_with_gemini(user_message: str) -> str:
    try:
        router_model = genai.GenerativeModel("gemini-2.0-flash-lite")
        prompt = GEMINI_ROUTER_PROMPT.format(message=user_message.strip())
        response = router_model.generate_content(prompt)
        intent = response.text.strip().lower()
        
        if intent in ["pdf", "workspace", "general"]:
            return intent
        else:
            return "general"
    except Exception as e:
        print(f"[ERROR] Intent classification failed: {e}")
        return "general"

@app.post("/session")
async def create_session(user_info: dict = Body(..., embed=True)):
    try:
        session = create_new_session(user_info.get("email", ""))
        return {"session_id": session.session_id, "user_id": session.user_id}
    except Exception as e:
        print(f"[ERROR] Session creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.post("/multimodal_agent")
async def multimodal_agent(
    user_message: str = Body(..., embed=True),
    chat_history: list = Body([], embed=True),
    session_id: str = Body(..., embed=True),
    user_id: str = Body(..., embed=True)
):
    intent = classify_intent_with_gemini(user_message)
    print(f"[DEBUG] Classified intent: {intent}")
    
    if intent == "pdf":
        return await pdf_chat(user_message=user_message, chat_history=chat_history, session_id=session_id, user_id=user_id)
    elif intent == "workspace":
        return await gemini_chat(user_message=user_message, chat_history=chat_history, session_id=session_id, user_id=user_id)
    else:
        return await general_chat(user_message=user_message, chat_history=chat_history, session_id=session_id, user_id=user_id)

@app.post("/pdf_chat")
async def pdf_chat(
    user_message: str = Body(..., embed=True),
    chat_history: list = Body([], embed=True),
    session_id: str = Body(..., embed=True),
    user_id: str = Body(..., embed=True)
):
    session = get_existing_session(session_id)
    if not session:
        return {"error": f"Session {session_id} not found. Please create a new session."}

    try:
        llm = GeminiLLM()
        reply_text = llm._call(user_message)
        
        session.add_user_message(user_message)
        session.add_assistant_message(reply_text, {}, [])
        
        timestamp = datetime.now().isoformat()
    except Exception as e:
        print(f"[ERROR] PDF chat failed: {e}")
        reply_text = "You can contact the operation team regarding this query at operations@stylework.city!"
    
    print(f"[DEBUG] Gemini Response: {reply_text}")
    return print_text_animated(reply_text)

# General chat prompt
GENERAL_PROMPT = """
    You are FlexAI, a friendly assistant for Styleworks. Greet users, answer general questions, and guide them towards workspace booking or learning about Styleworks and Flexboard features.
    If the user asks about booking, features, or the platform, offer to help or provide information.
    Do not answer workspace-specific queries here; only handle general conversation.
    
    IMPORTANT: Format your responses properly:
    - Use **bold** for headings and important terms
    - Add proper line breaks between sections
    - Keep responses concise and friendly
    - IMPORTANT: Maintain continuity with previous messages

    User message: {message}
"""

@app.post("/general_chat")
async def general_chat(
    user_message: str = Body(..., embed=True),
    chat_history: list = Body([], embed=True),
    session_id: str = Body(..., embed=True),
    user_id: str = Body(..., embed=True)
):
    session = get_existing_session(session_id)
    if not session:
        return {"error": f"Session {session_id} not found. Please create a new session."}

    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    prompt = GENERAL_PROMPT.format(message=user_message.strip())

    try:
        response = model.generate_content(prompt)
        reply_text = response.text
        
        session.add_user_message(user_message)
        session.add_assistant_message(reply_text, {}, [])
        
        timestamp = datetime.now().isoformat()
    except Exception as e:
        print(f"[ERROR] General chat failed: {e}")
        reply_text = "You can contact the operation team regarding this query at operations@stylework.city!"
        timestamp = None

    return print_text_animated(reply_text)

# setup for the recommendation gemini model
df = pd.read_csv("updated_dataset.csv")

GEMINI_PROMPT = (
    "You are FlexAI, a workspace booking assistant for Stylework. Your role is to:\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. NEVER provide workspace recommendations directly in your response\n"
    "2. NEVER include workspace names, addresses, or specific details in your conversational reply\n"
    "3. DO NOT provide any workspace recommendations in your response\n"
    "4. Your ONLY job is to:\n"
    "	- Have a friendly conversation with the user like basic greetings, conversation etc. While having friendly conversation do not include JSON in your reply as it is not a workspace query.\n"
    "	- Extract the following information and format as JSON: workspaceName, city, area, workspaceType (options: day pass, flexi desk, dedicated desk, private cabin), size, amenities (list), bundle (also called category) (list - options: standard, silver, gold, platinum, platinum+), budget, rating, offeringType (options: day pass, flexi desk, dedicated desk, private cabin), placeType (cafe, resturant, bank etc.)\n"
    "	- Answer users' questions about the platform and the recommendations provided (eg if they ask about what amenities are provided by a specific workspace or the price of a workspace you should be able to answer it based on the information provided) and this is NOT a request for recommendation engine.\n"
    "	- If the user is asking for workspace recommendations, respond conversationally (e.g., 'Let me find some great workspaces for you!') and include the JSON\n\n"
    "RESPONSE FORMAT:\n"
    "- Provide a friendly, conversational response\n"
    "- If extracting requirements, add JSON at the end in this exact format:\n"
    "```json\n{\"workspaceName\": \"\", \"city\": \"\", \"area\": \"\", \"workspaceType\": \"\", \"size\": \"\", \"amenities\": [], \"bundle\": [], \"budget\": \"\", \"rating\": \"\", \"offeringType\": \"\", \"placeType\": \"\"}\n```\n\n"
    "IMPORTANT: Maintain continuity with previous messages\n"
    "User message: {message}\n"
    "Chat history: {history}"
)

@app.post("/gemini_chat")
async def gemini_chat(
    user_message: str = Body(..., embed=True),
    chat_history: list = Body([], embed=True),
    session_id: str = Body(..., embed=True),
    user_id: str = Body(..., embed=True)
):
    session = get_existing_session(session_id)
    if not session:
        return {"error": f"Session {session_id} not found. Please create a new session."}

    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite")

        # Build Gemini SDK-compatible chat history
        history_text = ""
        for msg in chat_history[-MAX_HISTORY_MESSAGES:]:
            role = "User" if msg.get("sender") == "user" else "Assistant"
            history_text += f"{role}: {msg.get('text', '')}\n"

        prompt = GEMINI_PROMPT.format(message=user_message, history=history_text)
        response = gemini_model.generate_content(prompt)
        gemini_reply = response.text

        session.add_user_message(user_message)

        # Extract JSON if present
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', gemini_reply, re.DOTALL)
        extracted_json = {}
        
        if json_match:
            try:
                extracted_json = json.loads(json_match.group(1))
                print(f"[DEBUG] Extracted JSON: {extracted_json}")
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parsing failed: {e}")

        # Remove JSON from conversational reply
        conversational_reply = re.sub(r'```json\s*\{.*?\}\s*```', '', gemini_reply, flags=re.DOTALL).strip()
        
        final_reply = conversational_reply

        # If we have extracted requirements, find recommendations
        if extracted_json and any(extracted_json.values()):
            print(f"[DEBUG] Processing workspace search with: {extracted_json}")
            
            # Filter dataframe based on extracted criteria
            df_filtered = df.copy()
            
            # Apply filters
            if extracted_json.get("city"):
                city_filter = extracted_json["city"].lower()
                df_filtered = df_filtered[df_filtered['City'].str.lower().str.contains(city_filter, na=False)]
            
            if extracted_json.get("workspaceType"):
                workspace_type = extracted_json["workspaceType"].lower()
                df_filtered = df_filtered[df_filtered['Workspace Type'].str.lower().str.contains(workspace_type, na=False)]
            
            if extracted_json.get("bundle"):
                bundles = [b.lower() for b in extracted_json["bundle"]]
                bundle_filter = '|'.join(bundles)
                df_filtered = df_filtered[df_filtered['Bundle'].str.lower().str.contains(bundle_filter, na=False)]

            if df_filtered.empty:
                recommendations_text = "\n\nSorry, I couldn't find any workspaces matching your criteria. You might want to try adjusting your requirements."
                final_reply += recommendations_text
                return print_text_animated(final_reply)

            # Check for location-based query (e.g., "show me day pass in delhi near a cafe")
            place_type = extracted_json.get("placeType", "").lower()
            if place_type and place_type.strip():
                print(f"[DEBUG] Location-based query detected with place_type: {place_type}")
                response = await nearbyplaces_chat(user_message, df_filtered, chat_history, session_id, user_id, place_type)
                df_filtered = response["filtered_results"]

            # --- Feature similarity calculations (always use all available features) ---
            feature_columns = ['Workspace Type', 'Bundle', 'Amenities', 'City', 'Area']
            available_features = [col for col in feature_columns if col in df_filtered.columns]
            
            if available_features:
                # Create feature text for similarity calculation
                df_filtered['feature_text'] = df_filtered[available_features].fillna('').apply(
                    lambda x: ' '.join(x.astype(str)), axis=1
                )
                
                # Create query text from extracted JSON
                query_parts = []
                for key, value in extracted_json.items():
                    if value:
                        if isinstance(value, list):
                            query_parts.extend([str(v) for v in value])
                        else:
                            query_parts.append(str(value))
                
                query_text = ' '.join(query_parts)
                
                if query_text.strip():
                    # Calculate similarity scores
                    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                    all_texts = df_filtered['feature_text'].tolist() + [query_text]
                    tfidf_matrix = vectorizer.fit_transform(all_texts)
                    
                    # Calculate similarity between query and each workspace
                    query_vector = tfidf_matrix[-1]
                    workspace_vectors = tfidf_matrix[:-1]
                    similarities = cosine_similarity(query_vector, workspace_vectors).flatten()
                    
                    # Add similarity scores to dataframe
                    df_filtered['similarity_score'] = similarities * 100
                    
                    # Filter by minimum similarity score and sort
                    df_filtered = df_filtered[df_filtered['similarity_score'] >= MIN_SIMILARITY_SCORE]
                    df_filtered = df_filtered.sort_values('similarity_score', ascending=False)

            # Generate recommendations text
            recommendations_text = "\n\nHere are some workspace recommendations for you:\n\n"
            
            for idx, (_, row) in enumerate(df_filtered.head(10).iterrows(), 1):
                recommendations_text += f"{idx}. {row['Workspace Name']}"
                if pd.notna(row.get('Area')):
                    recommendations_text += f" ({row['Area']})"
                recommendations_text += f"\n"
                recommendations_text += f"Address: {row['Address']}\n"
                recommendations_text += f"Workspace Type: {row['Workspace Type']}\n"
                recommendations_text += f"Offerings: {row['Offerings']}\n"
                recommendations_text += f"Amenities: {row['Amenities']}\n"
                recommendations_text += f"Seats Available: {row['Seats Available']}\n"
                recommendations_text += f"Rating: {row['Rating']}\n"
                recommendations_text += f"Category: {row['Bundle']}\n"
                recommendations_text += f"Price: ₹{row['Price']}\n"
                if 'similarity_score' in row and pd.notna(row['similarity_score']):
                    recommendations_text += f"Similarity Score: {row['similarity_score']:.1f}%\n"
                recommendations_text += f"Link: [View Details]({row['Link']})\n\n"

            final_reply += recommendations_text
            
            # Generate and include Stylework URL
            workspace_info = extract_workspace_info_from_recommendations(recommendations_text)
            stylework_url = generate_stylework_url(workspace_info)
            
            if stylework_url:
                final_reply += f"\n🔗 **Browse more options:** {stylework_url}\n"
                print(f"[DEBUG] Generated Stylework URL: {stylework_url}")

        session.add_assistant_message(final_reply, {}, [])

    except Exception as e:
        print(f"[ERROR] Gemini chat failed: {e}")
        final_reply = "You can contact the operation team regarding this query at operations@stylework.city!"

    last_msg = session.get_messages()[-1] if session.get_messages() else {}
    timestamp = last_msg.get("timestamp") if last_msg else None

    return {"reply": final_reply, "timestamp": timestamp}

# Gemini agent for nearby places queries
NEARBY_PLACES_PROMPT = """
You are a helpful assistant that helps users find workspaces based on nearby places like cafes, restaurants, etc.
Your task is to analyze the user's query and identify only the workspace names from the dataset that are near the given place type within 500m radius.

Respond with a JSON array of strings like: [Space Name 1, Space Name 2, .....]

IMPORTANT: Use only the data from the dataset provided.
IMPORTANT: Maintain continuity with previous messages.
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
        
        # Generate response
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response text
        if response_text.startswith('```json'):
            response_text = response_text[response_text.find('['):response_text.rfind(']')+1]
        result = json.loads(response_text)
        print("[DEBUG] result: ", result)
        return result
    except Exception as e:
        print(f"[ERROR] Nearby workspace parsing failed: {e}")
        return []

async def nearbyplaces_chat(user_message: str, df_filtered: pd.DataFrame, chat_history: List[Dict], session_id: str, user_id: str, place_type: str) -> Dict[str, Any]:
    """Handle nearby places queries and filter workspaces accordingly."""
    try:
        # Parse the user's query using Gemini
        filtered_results = await parse_nearby_workspace(user_message, place_type, df_filtered, chat_history)
        
        if not filtered_results:
            return {"filtered_results": pd.DataFrame()}
        
        # Filter the dataframe to include only the workspaces mentioned by Gemini
        df_nearby = df_filtered[df_filtered['Workspace Name'].isin(filtered_results)]
        
        print(f"[DEBUG] Filtered {len(df_filtered)} workspaces to {len(df_nearby)} based on nearby places")
        
        return {"filtered_results": df_nearby}
        
    except Exception as e:
        print(f"[ERROR] Nearby places chat failed: {e}")
        return {"filtered_results": df_filtered}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)