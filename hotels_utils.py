from typing import List, Dict, Optional
import pandas as pd
from chromadb_lib import ChromaHotelSearch  

def init_hotel_index(
    chroma_path: str = "./chroma_storage",
    collection_name: str = "hotel_information",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> ChromaHotelSearch:
    return ChromaHotelSearch(
        chromadb_storage_path=chroma_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

def ingest_hotels(
    hotels: ChromaHotelSearch, 
    df_hotel: pd.DataFrame,
    country: str,
    city: str,
    country_col: str = " countyName",
    city_col: str = " cityName"     
) -> None:
    """
    Filters the DataFrame for a specific city/country BEFORE upserting.
    """
    # print(f"Original hotel DataFrame count: {len(df_hotel)}")
    filtered_df = df_hotel[
        (df_hotel[country_col] == country) & 
        (df_hotel[city_col] == city)
    ].copy()
    
    if filtered_df.empty:
        print(f"Warning: No hotels found for {city}, {country}. No data ingested.")
        return
        
    # print(f"Filtered count for {city}, {country}: {len(filtered_df)}")
    
    hotels.upsert_hotels_from_df(filtered_df)

def search_hotels(
    hotels: ChromaHotelSearch,
    *,
    prefs_text: str,
    top_n: int = 5,
    country: Optional[str] = None,
    city: Optional[str] = None,
    min_rating: Optional[float] = None,
    rating_labels: Optional[List[str]] = None,
    include_docs: bool = True,
) -> List[Dict]:
    return hotels.top_n_relevant(
        query=prefs_text,
        top_n=top_n,
        country=country,
        city=city,
        min_rating=min_rating,
        rating_in=rating_labels,
        include_docs=include_docs,
    )
