# hotels_utils.py
from typing import List, Dict, Optional
import pandas as pd
from chromadb_lib import ChromaHotelSearch  # your earlier class file

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

def ingest_hotels(hotels: ChromaHotelSearch, df_hotel: pd.DataFrame) -> None:
    hotels.upsert_hotels_from_df(df_hotel)

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
