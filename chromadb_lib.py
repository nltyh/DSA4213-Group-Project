import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any
import pandas as pd
import math

class ChromaHotelSearch:
    """
    - Uses a persistent Chroma collection.
    - Embeds hotel text (name + description + attractions + facilities).
    - Stores structured fields as metadata for fast filtering.
    """

    def __init__(
        self,
        chromadb_storage_path: str,
        collection_name: str = "hotel_information",
        embedding_model: str = "all-MiniLM-L6-v2",
        create_if_missing: bool = True,
    ):
        self.client = chromadb.PersistentClient(path=chromadb_storage_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        # Create or get collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception:
            if not create_if_missing:
                raise
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )

    # ------------ Public API ------------

    def upsert_hotels_from_df(
        self,
        df: pd.DataFrame,
        id_col: str = " HotelCode",
        country_col: str = " countyName",  # your CSV uses countyName/countyCode for country
        city_col: str = " cityName",
        rating_col: str = " HotelRating",
        name_col: str = " HotelName",
        desc_col: str = " Description",
        attr_col: str = " Attractions",
        fac_col: str = " HotelFacilities",
        addr_col: str = " Address",
        website_col: str = " HotelWebsiteUrl",
        batch_size: int = 512,
    ):
        """
        Build a clean document string to embed, and store the rest as metadata.
        """
        ids, docs, metas = [], [], []

        for _, r in df.iterrows():
            # Build document (what we embed)
            doc = self._build_document(
                name=r.get(name_col),
                description=r.get(desc_col),
                attractions=r.get(attr_col),
                facilities=r.get(fac_col),
                city=r.get(city_col),
                country=r.get(country_col),
                address=r.get(addr_col)
            )

            # Parse rating as a normalized categorical/numeric
            rating_val = self._normalize_rating(r.get(rating_col))

            meta = {
                "hotel_code": self._safe_str(r.get(id_col)),
                "country": self._safe_str(r.get(country_col)),
                "city": self._safe_str(r.get(city_col)),
                "rating": rating_val,  # e.g., 4.0 for "FourStar"
                "rating_raw": self._safe_str(r.get(rating_col)),
                "name": self._safe_str(r.get(name_col)),
                "address": self._safe_str(r.get(addr_col)),
                "website": self._safe_str(r.get(website_col))
            }

            ids.append(f"hotel:{meta['hotel_code']}")
            docs.append(doc)
            metas.append(meta)

            # Batched upserts to keep memory in check
            if len(ids) >= batch_size:
                self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
                ids, docs, metas = [], [], []

        if ids:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def top_n_relevant(
        self,
        query: str,
        top_n: int = 5,
        country: Optional[str] = None,
        city: Optional[str] = None,
        min_rating: Optional[float] = None,   # e.g., 4.0 to mean >= 4 stars
        rating_in: Optional[List[str]] = None,  # filter by raw labels (e.g., ["FourStar","FiveStar"])
        extra_where: Optional[Dict[str, Any]] = None,  # pass through extra Chroma where filters
        include_docs: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Query with semantic text + structured metadata filters.

        Returns a list of:
        {
            "hotel_code": ...,
            "name": ...,
            "country": ...,
            "city": ...,
            "rating": 4.0,
            "rating_raw": "FourStar",
            "distance": 0.18,   # cosine distance (lower is better)
            "document": "...",  # optional
            "id": "hotel:1002140"
        }
        """
        where_and = []

        if country:
            where_and.append({"country": country.strip()})
        if city:
            where_and.append({"city": city.strip()})
        if min_rating is not None:
            where_and.append({"rating": {"$gte": float(min_rating)}})
        if rating_in:
            where_and.append({"rating_raw": {"$in": rating_in}})

        if extra_where:
            # Caller can pass any Chroma-compatible filter dict; we AND it in
            where_and.append(extra_where)

        where_clause = {"$and": where_and} if where_and else {}

        include_fields = ["metadatas", "distances"]
        if include_docs:
            include_fields.append("documents")

        res = self.collection.query(
            query_texts=[query],
            n_results=top_n,
            where=where_clause,
            include=include_fields
        )

        # Chroma returns lists-of-lists; take first query's result
        docs = res.get("documents", [[]])[0] if include_docs else [None] * len(res["ids"][0])
        metas = res["metadatas"][0] if "metadatas" in res else [{}] * len(res["ids"][0])
        ids = res["ids"][0]
        dists = res["distances"][0] if "distances" in res else [None] * len(ids)

        out = []
        for i in range(len(ids)):
            m = metas[i] or {}
            item = {
                "id": ids[i],
                "hotel_code": m.get("hotel_code"),
                "name": m.get("name"),
                "country": m.get("country"),
                "city": m.get("city"),
                "rating": m.get("rating"),
                "rating_raw": m.get("rating_raw"),
                "distance": dists[i],
            }
            if include_docs:
                item["document"] = docs[i]
            out.append(item)
        return out

    # ------------ Helpers ------------

    @staticmethod
    def _build_document(
        name: Optional[str],
        description: Optional[str],
        attractions: Optional[str],
        facilities: Optional[str],
        city: Optional[str],
        country: Optional[str],
        address: Optional[str],
    ) -> str:
        parts = []
        if name:
            parts.append(str(name).strip())
        # brief header with location
        loc_bits = [b for b in [city, country] if pd.notna(b) and str(b).strip()]
        if loc_bits:
            parts.append(" — " + ", ".join(loc_bits))
        if address and str(address).strip():
            parts.append(f"\nAddress: {address.strip()}")
        if description and str(description).strip():
            parts.append(f"\nSummary: {str(description).strip()}")
        if attractions and str(attractions).strip():
            parts.append(f"\nNearby: {str(attractions).strip()}")
        if facilities and str(facilities).strip():
            parts.append(f"\nFacilities: {str(facilities).strip()}")
        return "".join(parts) if parts else ""

    @staticmethod
    def _normalize_rating(raw: Any) -> Optional[float]:
        """
        Converts strings like "FourStar" to 4.0.
        Leaves numeric as float.
        """
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return None
        s = str(raw).strip()
        # direct numeric?
        try:
            return float(s)
        except ValueError:
            pass
        # simple mapping
        mapping = {
            "OneStar": 1.0, "TwoStar": 2.0, "ThreeStar": 3.0,
            "FourStar": 4.0, "FiveStar": 5.0,
            "1 Star": 1.0, "2 Star": 2.0, "3 Star": 3.0, "4 Star": 4.0, "5 Star": 5.0,
            "1-star": 1.0, "2-star": 2.0, "3-star": 3.0, "4-star": 4.0, "5-star": 5.0,
        }
        return mapping.get(s, None)

    @staticmethod
    def _safe_str(x: Any) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        return s if s else None


