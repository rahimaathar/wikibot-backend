from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import cachetools
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self):
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Cache for storing embeddings
        self.embedding_cache = cachetools.TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        # Store texts and their embeddings
        self.texts = []
        self.embeddings = []
    
    def get_embedding(self, text: str) -> np.ndarray:
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        # Cache the embedding
        self.embedding_cache[text] = embedding
        return embedding
    
    def add_texts(self, texts: List[str]):
        """Add texts to the search index."""
        # Generate embeddings for all texts
        embeddings = [self.get_embedding(text) for text in texts]
        
        # Add to storage
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar texts using the query."""
        if not self.texts:
            return []
            
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return results with texts and similarities
        results = []
        for idx in top_indices:
            results.append((self.texts[idx], float(similarities[idx])))
        
        return results
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return float(cosine_similarity([emb1], [emb2])[0][0])

# Create a singleton instance
semantic_search = SemanticSearch() 