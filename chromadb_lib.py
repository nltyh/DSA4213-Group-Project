import chromadb 
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional

class chromaQuery:
    def __init__(self,chromadb_storage_path: str,collection_name: str = 'hotel_information',embedding_model: str = 'all-MiniLM-L6-v2'):
        self.client = chromadb.PersistentClient(path = chromadb_storage_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = embedding_model)
        self.collection = self.client.get_collection(name = collection_name)
    
    def top_n_relevant(self, query: str, top_n: int, country: Optional[str] = None, city: Optional[str] = None) -> List[Dict]:
        final = []
        where_clause = {}
        filters = []
        if country:
            filters.append({'country': country.title().strip()})
        if city:
            filters.append({'city': city.title().strip()})
        if filters:
            where_clause = {'$and': filters}
        
        results = self.collection.query(
            query_texts = [query],
            n_results = top_n,
            where = where_clause,
            include = ['documents', 'metadatas']
        )

        # cleaning the chromadb output, can edit as needed
        documents_retrieved = results['documents'][0]
        metadatas_retrieved = results['metadatas'][0]

        for i in range(len(documents_retrieved)):
            d = {
                'hotel_code': metadatas_retrieved[i]['hotel code'],
                'hotel_info': documents_retrieved[i]
            }
            final.append(d)
        
        return final

"""
# initialise the class
chroma = chromaQuery(
    chromadb_storage_path = "./chroma_storage"
)

# query
results = chroma.top_n_relevant(
    query = "luxury hotels with pool",
    top_n = 5,
    country = "Japan", #optional
    city = "Tokyo" #optional
)

# output (length == top_n)
# info is the attractions, descriptions and facilities
# need to cross reference hotel code with hotels.csv
[
    {'hotel_code': ___, 'hotel_info': _____},
    {'hotel_code': ___, 'hotel_info': _____}
]
"""

