import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import date, datetime
from typing import Optional, Literal
from pydantic import BaseModel, ValidationError
import json

# Import utilities
import chromadb_lib as cdb
import hotels_utils as hu
import flight_api as fa

# -- Streamlit UI --
st.set_page_config(page_title=" RoamEasy: AI Travel Assistant", layout="wide")
st.title("🤖 RoamEasy: AI Travel Assistant")
st.caption("Enter your travel plans with exact dates and locations, and I'll find flights and hotels.")


# -- Load Data and Configure Generative AI --

GEMINI_API_KEY = "AIzaSyBNHTOBkzNbpcywUrTWruS3d_wFKTayPqA"
genai.configure(api_key=GEMINI_API_KEY)

# --- 3. Caching (for performance) ---

@st.cache_data
def load_data():
    """Loads dataframes. Creates mock data if files are missing."""
    try:
        flights_df = pd.read_csv("flights_data.csv")
    except FileNotFoundError:
        st.warning("flights_data.csv not found. Using mock flight data.")
        flights_df = pd.DataFrame({
            'origin_iata': ['ZRH', 'LHR'], 'destination_iata': ['FCO', 'JFK'],
            'departure_date': ['2025-11-10 18:00:00', '2025-12-01 09:00:00'],
            'arrival_date': ['2025-11-10 19:30:00', '2025-12-01 12:00:00'],
            'return_departure_date': ['2025-11-24 20:00:00', '2025-12-10 18:00:00'],
            'return_arrival_date': ['2025-11-24 21:30:00', '2025-12-11 07:00:00'],
            'airline': ['Swiss', 'British Airways'], 'price': [350.00, 600.00],
            'currency': ['USD', 'USD'], 'fare_type': ['ECONOMY', 'BUSINESS'],
            'passenger_breakdown': ['A1 C0', 'A2 C0']
        })
    
    try:
        hotels_df = pd.read_parquet("hotels.parquet")
    except FileNotFoundError:
        st.warning("hotels.parquet not found. Using mock hotel data.")
        hotels_df = pd.DataFrame({
            ' HotelCode': ["101", "102"],
            ' countyName': ["Italy", "United States"],
            ' cityName': ["Rome", "New York,   NY"],
            ' HotelRating': ["FiveStar", "FourStar"],
            ' HotelName': ["Hotel Roma", "The Big Apple Hotel"],
            ' Description': ["A lovely hotel in the center of Rome.", "Great views of the city."],
            ' Attractions': ["Colosseum", "Times Square"],
            ' HotelFacilities': ["Wifi, Pool, Spa", "Wifi, Gym, Restaurant"],
            ' Address': ["123 Via Roma", "456 5th Ave"],
            ' HotelWebsiteUrl': ["http://hotelroma.com", "http://bighotel.com"]
        })
    return flights_df, hotels_df

# -- Pydantic Model for Trip Query --

TimeWindow = Literal["morning","afternoon","evening","night"]

class TripQuery(BaseModel):
    origin: str
    destination: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    adults: int = 1
    children: int = 0
    cabin: Literal["ECONOMY","PREMIUM_ECONOMY","BUSINESS","FIRST"] = "ECONOMY"
    depart_window: Optional[TimeWindow] = None
    arrive_window: Optional[TimeWindow] = None
    return_window: Optional[TimeWindow] = None
    hotel_country: Optional[str] = None
    hotel_city: Optional[str] = None
    hotel_min_rating: Optional[float] = None
    hotel_prefs_text: Optional[str] = None
    
    
@st.cache_resource
def init_models_and_index():
    """Initializes and caches the GenAI models and the ChromaDB index."""
    extractor_model = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config={"response_mime_type": "application/json"}
    )
    
    agent_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT
    )
    
    hotels_index = hu.init_hotel_index("./chroma_storage", "hotel_information", "all-MiniLM-L6-v2")
    
    return extractor_model, agent_model, hotels_index

# -- Generative AI-based Trip Query Extractor --
def extractor(model: genai.GenerativeModel, text: str) -> Optional[TripQuery]:
    schema = TripQuery.model_json_schema()
    # The prompt provides the context, instructions, the schema, and the user text.
    # This guides the LLM to perform the extraction task accurately.
    prompt = f"""
    You are an expert travel assistant responsible for extracting structured data from user requests.
    Your goal is to parse the user's text and output a JSON object that strictly adheres to the provided schema.

    CONTEXT:
    - Today's date is: {datetime.now().strftime('%Y-%m-%d')}
    - Time window definitions: Departures between 6am-12pm are "morning", 12pm-5pm are "afternoon", 5pm-9pm are "evening", and 9pm-6am are "night".

    INSTRUCTIONS:
    1.  Analyze the user's text to extract all relevant travel details.
    2.  For `origin` and `destination`, extract the most popular IATA codes like ZRH for Zurich, FCO for Rome, etc. unless the user specifies otherwise.
    4.  Use the provided mappings to normalize values. For example, if they say "business class", use "BUSINESS".
    5.  If a value is not mentioned in the text, omit it or set it to null in the JSON.
    6.  Infer `hotel_city` and `hotel_country` from the main destination. If it is in the USA, write it as United States, and the city as [City],   [State] (the 3 blank spaces are intentional). For example: 'Abbeville,   Louisiana'
    7.  The `hotel_prefs_text` field should contain the original, unmodified user text.
    8.  Parse dates accurately. "Next Tuesday" should be calculated relative to today's date.
    7.  Your output MUST be a valid JSON object matching the schema below.

    SCHEMA:
    {json.dumps(schema, indent=2)}

    USER TEXT:
    "{text}"
    """

    try:
        response = model.generate_content(prompt)
        json_data = json.loads(response.text)
        
        # Use Pydantic to validate the JSON and create the TripQuery object
        trip_query = TripQuery(**json_data)
        return trip_query

    except json.JSONDecodeError:
        print("Error: The model did not return valid JSON.")
        print("Model output:", response.text)
        return None
    except ValidationError as e:
        print(f"Error: Pydantic validation failed.\n{e}")
        print("Model output:", response.text)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
def hotels_context(hits)->str:
    if not hits:
        return "No hotel options matched the filters."
    lines=[]
    for h in hits:
        rating = f"{h.get('rating'):.0f}★" if h.get('rating') else (h.get('rating_raw') or "N/A")
        addr = h.get("address") or ""
        url  = h.get("website") or ""
        desc = h.get("description") or "No description available."
        if len(desc) > 200:
            desc = desc 
        fac = h.get("facilities") or "No facilities listed."
        if len(fac) > 150:
            fac = fac
        lines.append(
            f"* {h.get('name','N/A')} — {h.get('city','')}, {h.get('country','')}\n"
            f"  Rating: {rating}\n"
            f"  Address: {addr}\n"
            f"  Website: {url}\n"
            f"  Description: {desc}\n"
            f"  Facilities: {fac}"
        )
    return "\n".join(lines)

# System Prompt and Format Rules for the Travel Agent Model
SYSTEM_PROMPT = """You are a meticulous travel planning assistant.
Use only the context blocks provided for:
- Flight details
- Hotel options
Do not fabricate or assume details not present in the context.

When responding:
1) Extract key info (origin, destination, dates, travelers, budget/class preferences).
2) If anything is missing, ask the user for it.
3) Filter and pick the best options from the context according to the user's ask e.g. (cheapest flight).
4) Output EXACTLY in the required format."""

FORMAT_RULES = """
Output Format:

Flights (Round Trip)
* Outbound (<origin> → <destination>)
  <airline(s)> — <origin> <departure date/time> → <destination> <arrival date/time> | <currency> <price> | <passenger breakdown> <class>
* Return (<destination> → <origin>)
  <airline(s)> — <destination> <departure date/time> → <origin> <arrival date/time>

Hotel
* Hotel Name: <hotel name>
* Rating: <star rating>
* Address: <address>
* Website: <URL>
* Description: <short description> summarised
* Facilities: <list of facilities> summarised

Suggested Itinerary:
* Feel free to suggest a brief itinerary based on the flight times and hotel location.

Formatting Rules:
- Passenger breakdown: A = Adult, C = Child (e.g., A2 C1).
- Write the full class type (e.g., “ECONOMY”).
- Dates/times format: “DD Mon YYYY HH:MM”.
"""

# -- Main Workflow --
flights_df, hotels_df = load_data()
extractor_model, agent_model, hotels_index = init_models_and_index()

user_input = st.text_area(
    "Your travel query:",
    "I need a trip from Zurich to Rome for 2 adults and 1 child, leaving next Monday and returning in two weeks. We'll fly economy. I need a 4-star or better hotel, preferably near the Vatican.",
    height=150
)

if st.button("Plan My Trip"):
    if not user_input:
        st.warning("Please enter your travel plans.")
        st.stop()

    trip = None
    with st.spinner("Analyzing your request..."):
        trip = extractor(extractor_model, user_input)
    
    if not trip:
        st.error("I couldn't understand your request. Please try rephrasing.")
        st.stop()

    st.success("Request analyzed. Here's the structured query:")
    st.json(trip.model_dump_json(indent=2))
    
    # --- Hotel Ingestion ---
    with st.spinner(f"Loading hotel data for {trip.hotel_city}..."):
        # This pre-filters and upserts only relevant hotels
        hu.ingest_hotels(
            hotels_index, 
            hotels_df, 
            country=trip.hotel_country, 
            city=trip.hotel_city
        )

    # --- Flight Search ---
    
    #static data frame for testing
    flight_context = ""
    with st.spinner("Searching for flights..."):
        '''
        parsed_flights = fa.parse_flights_df(flights_df, summary_col="summary")
        flight_results = fa.pick_flight(
            parsed_flights,
            origin=trip.origin,
            destination=trip.destination,
            start_date=trip.start_date,
            end_date=trip.end_date,
            fare_type=trip.cabin
        )
        flight_context = fa.flights_context(flight_results)
        '''
        # Flights api call alternative:
        flight_results= fa.search_flights_api(trip)
        flight_context = "\n".join(flight_results)

    # --- Hotel Search ---
    hotel_context_str = ""
    with st.spinner("Searching for hotels..."):
        hotel_hits = hu.search_hotels(
            hotels_index,
            prefs_text=trip.hotel_prefs_text,
            top_n=5,
            country=trip.hotel_country,
            city=trip.hotel_city,
            min_rating=trip.hotel_min_rating or 4.0,
        )
    # --- Final AI Generation ---
    with st.spinner("Generating your itinerary..."):
        CTX = f"""[FLIGHTS] 
        {flight_context} 
        [HOTELS] 
        {hotels_context(hotel_hits)} """

    # -- Generative AI-based Travel Agent Response Generation --
        user_task = f"""User Message:
        {user_input}

        Context:
        {CTX}

        Follow the Output Format exactly. If any required field is missing in context, state what is missing and ask the user for it (instead of guessing)."""

        try:
            resp = agent_model.generate_content([FORMAT_RULES, user_task])
            
            st.success("Your trip plan is ready!")
            st.markdown(resp.text)
            
            with st.expander("Show Raw Context Sent to AI"):
                st.code(CTX, language="markdown")
                
        except Exception as e:
            st.error(f"Error generating response: {e}")
