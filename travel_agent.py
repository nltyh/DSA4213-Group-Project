import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import date, datetime
from typing import Optional, Literal
from pydantic import BaseModel, ValidationError
import json

# Import utilities
from main_workflow import *

# -- Streamlit UI --
st.set_page_config(page_title=" RoamEasy: AI Travel Assistant", layout="wide")
st.title("🤖 RoamEasy: AI Travel Assistant")
st.caption("Enter your travel plans with exact dates and locations, and I'll find flights and hotels.")


# -- Load Data and Configure Generative AI --

GEMINI_API_KEY = "AIzaSyBNHTOBkzNbpcywUrTWruS3d_wFKTayPqA"
genai.configure(api_key=GEMINI_API_KEY)

def parse_hotels_for_display(ctx: str) -> pd.DataFrame:
    """
    Extracts hotel info from the [HOTELS] section of CTX into a DataFrame.
    Columns: Name, City, Country, Rating, Address, Website, Description, Facilities
    """
    hotels_df = []
    
    # Extract the [HOTELS] section
    hotels_section = re.search(r"\[HOTELS\](.*)", ctx, re.DOTALL)
    if not hotels_section:
        return pd.DataFrame()
    
    hotels_text = hotels_section.group(1).strip()
    
    # Split by '* ' which indicates a new hotel entry
    hotel_entries = hotels_text.split("* ")
    
    for entry in hotel_entries:
        if not entry.strip():
            continue
        lines = entry.strip().split("\n")
        name_city_country = lines[0].split(" — ")
        if len(name_city_country) != 2:
            continue
        name = name_city_country[0].strip()
        city_country = name_city_country[1].split(",")
        city = city_country[0].strip() if len(city_country) > 0 else ""
        country = city_country[1].strip() if len(city_country) > 1 else ""
        
        # Extract other fields
        rating = ""
        address = ""
        website = ""
        desc = ""
        facilities = ""
        
        for line in lines[1:]:
            if line.startswith("  Rating:"):
                rating = line.replace("Rating:", "").strip()
            elif line.startswith("  Address:"):
                address = line.replace("Address:", "").strip()
            elif line.startswith("  Website:"):
                website = line.replace("Website:", "").strip()
            elif line.startswith("  Description:"):
                desc = line.replace("Description:", "").strip()
            elif line.startswith("  Facilities:"):
                facilities = line.replace("Facilities:", "").strip()
        
        hotels_df.append({
            "Name": name,
            "City": city,
            "Country": country,
            "Rating": rating,
            "Address": address,
            "Website": website,
            "Description": desc,
            "Facilities": facilities
        })
        
    return pd.DataFrame(hotels_df)

def parse_flights_for_display(ctx: str) -> pd.DataFrame:
    """
    Extracts flight info from the [FLIGHTS] section of CTX into a DataFrame.
    Columns: Origin, Destination, Airline, Departure, Arrival, Price, Currency, Passenger, Class
    """
    flights_df = []
    
    # Extract the [FLIGHTS] section
    flights_section = re.search(r"\[FLIGHTS\](.*?)\n\[HOTELS\]", ctx, re.DOTALL)
    if not flights_section:
        return pd.DataFrame()  # empty if not found
    
    flights_text = flights_section.group(1).strip()
    # Split each flight (assume starts with '* Outbound' or '* Return')
    flight_entries = re.findall(r"\* (Outbound|Return) \((.*?) → (.*?)\)\n\s*(.*?)$", flights_text, re.MULTILINE)
    
    for entry in flight_entries:
        direction, origin, destination, details = entry
        # Parse details line: airline — origin dep → dest arr | currency price | passengers class
        m = re.match(r"(.*?) — .*? (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) → .*? (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) \| (\w+) (\d+\.?\d*) \| (.*?) (.*)", details)
        if m:
            airline, dep, arr, currency, price, passengers, cabin = m.groups()
            flights_df.append({
                "Direction": direction,
                "Origin": origin,
                "Destination": destination,
                "Airline": airline.strip(),
                "Departure": dep,
                "Arrival": arr,
                "Currency": currency,
                "Price": float(price),
                "Passenger": passengers,
                "Class": cabin
            })
    return pd.DataFrame(flights_df)

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
        trip = extractor(user_input, extractor_model)
    
    if not trip:
        st.error("I couldn't understand your request. Please try rephrasing.")
        st.stop()
    
    with st.spinner("Generating your itinerary..."):
        res, ctx = run_travel_bot(user_input)
        st.success("Your trip plan is ready!")

        # --- Display AI-generated suggested itinerary ---
        st.subheader("🗺 Suggested Itinerary")
        st.markdown(res)

        # --- Flights and Hotels in tables ---
        st.subheader("✈ All Flights Found")
        try:
            # Convert flight context string back to a DataFrame for display
            flights_df_display = parse_flights_for_display(ctx)  
            if len(flights_df_display) == 0:
                st.info("No flight found!")
            else:
                st.dataframe(flights_df_display)
        except Exception:
            st.info("Flight details not available in table format; see AI-generated itinerary above.")

        st.subheader("🏨 All Hotels Found")
        try:
            hotels_df_display = parse_hotels_for_display(ctx)  
            if len(hotels_df_display) == 0:
                st.info('No hotels found!')
            else:
                st.dataframe(hotels_df_display)
        except Exception:
            st.info("Hotel details not available in table format; see AI-generated itinerary above.")

        # --- Raw context for debugging / transparency ---
        with st.expander("Show Raw Context Sent to AI"):
            st.code(ctx, language="markdown")


    # st.success("Request analyzed. Here's the structured query:")
    # st.json(trip.model_dump_json(indent=2))
    
    # --- Hotel Ingestion ---
    # with st.spinner(f"Loading hotel data for {trip.hotel_city}..."):
    #     # This pre-filters and upserts only relevant hotels
    #     hu.ingest_hotels(
    #         hotels_index, 
    #         hotels_df, 
    #         country=trip.hotel_country, 
    #         city=trip.hotel_city
    #     )

    # --- Flight Search ---
    
    #static data frame for testing
    # flight_context = ""
    # with st.spinner("Searching for flights..."):
    #     parsed_flights = fa.parse_flights_df(flights_df, summary_col="summary")
    #     flight_results = fa.pick_flight(
    #         parsed_flights,
    #         origin=trip.origin,
    #         destination=trip.destination,
    #         start_date=trip.start_date,
    #         end_date=trip.end_date,
    #         fare_type=trip.cabin
    #     )
    #     flight_context = fa.flights_context(flight_results)
        # Flights api call alternative:
        # flight_results= fa.search_flights_api(trip)
        # flight_context = "\n".join(flight_results)

    # --- Hotel Search ---
    # hotel_context_str = ""
    # with st.spinner("Searching for hotels..."):
    #     hotel_hits = hu.search_hotels(
    #         hotels_index,
    #         prefs_text=trip.hotel_prefs_text,
    #         top_n=5,
    #         country=trip.hotel_country,
    #         city=trip.hotel_city,
    #         min_rating=trip.hotel_min_rating or 4.0,
    #     )
    # --- Final AI Generation ---
    # with st.spinner("Generating your itinerary..."):
    #     CTX = f"""[FLIGHTS] 
    #     {flight_context} 
    #     [HOTELS] 
    #     {hotels_context(hotel_hits)} """

    # # -- Generative AI-based Travel Agent Response Generation --
    #     user_task = f"""User Message:
    #     {user_input}

    #     Context:
    #     {CTX}

    #     Follow the Output Format exactly. If any required field is missing in context, state what is missing and ask the user for it (instead of guessing)."""

    #     try:
    #         resp = agent_model.generate_content([FORMAT_RULES, user_task])
            
    #         st.success("Your trip plan is ready!")
    #         st.markdown(resp.text)
            
    #         with st.expander("Show Raw Context Sent to AI"):
    #             st.code(CTX, language="markdown")
                
    #     except Exception as e:
    #         st.error(f"Error generating response: {e}")