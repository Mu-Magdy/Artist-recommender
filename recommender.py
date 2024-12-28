import pandas as pd
from typing import List, Dict
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path

class ArtistRecommender:
    def __init__(self, index_path: str = 'artist_index'):
        # Initialize the sentence transformer model for text embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scaler = MinMaxScaler()
        self.index_path = Path(index_path)
        
        # Initialize FAISS index
        self.dimension = 384  # dimension of all-MiniLM-L6-v2 embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store artist metadata
        self.artists_data = []
        
    def prepare_artist_data(self, artists_df: pd.DataFrame) -> None:
        """
        Process artist data and create embeddings for vector storage
        """
        # Reset index to make sure we have sequential IDs
        artists_df = artists_df.reset_index(drop=True)
        
        # Normalize numerical values (subscribers)
        self.scaler.fit(artists_df[['subscribers']].values)
        
        # Create text descriptions for each artist
        artists_df['description'] = artists_df.apply(
            lambda x: f"Artist {x['name']} from {x['country']} creates {x['genre']} music "
                     f"with {x['subscribers']} subscribers", axis=1
        )
        
        # Generate embeddings for each artist
        embeddings = self.model.encode(artists_df['description'].tolist())
        
        # Convert to numpy array and add to FAISS index
        embeddings_np = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)
        
        # Store artist metadata
        self.artists_data = artists_df.to_dict('records')
        
        # Save index and metadata
        self._save_index()

    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        # Create directory if it doesn't exist
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path / 'artists.index'))
        
        # Save metadata
        with open(self.index_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(self.artists_data, f)

    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        if (self.index_path / 'artists.index').exists():
            self.index = faiss.read_index(str(self.index_path / 'artists.index'))
            with open(self.index_path / 'metadata.pkl', 'rb') as f:
                self.artists_data = pickle.load(f)
            return True
        return False

    def get_recommendations(self, campaign_requirements: str, top_k: int = 5) -> List[Dict]:
        """
        Get artist recommendations based on campaign requirements
        """
        # Generate embedding for campaign requirements
        query_embedding = self.model.encode(campaign_requirements)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        recommendations = []
        for idx, (distance, index) in enumerate(zip(distances[0], indices[0])):
            if index < len(self.artists_data):  # Check if index is valid
                artist_data = self.artists_data[index]
                recommendations.append({
                    'artist': artist_data['name'],
                    'country': artist_data['country'],
                    'genre': artist_data['genre'],
                    'subscribers': artist_data['subscribers'],
                    'similarity_score': 1 / (1 + distance)  # Convert distance to similarity score
                })
        
        return recommendations

    def filter_recommendations(self, 
                             recommendations: List[Dict],
                             min_subscribers: int = None,
                             countries: List[str] = None,
                             genres: List[str] = None) -> List[Dict]:
        """
        Apply additional filters to recommendations
        """
        filtered = recommendations.copy()
        
        if min_subscribers:
            filtered = [r for r in filtered if r['subscribers'] >= min_subscribers]
            
        if countries:
            filtered = [r for r in filtered if r['country'] in countries]
            
        if genres:
            filtered = [r for r in filtered if r['genre'] in genres]
            
        return filtered

# # Example usage
# if __name__ == "__main__":
#     # Sample data
#     data = {
#         'name': ['Artist1', 'Artist2', 'Artist3'],
#         'country': ['USA', 'UK', 'Canada'],
#         'subscribers': [1000000, 500000, 750000],
#         'genre': ['Pop', 'Rock', 'Hip-Hop']
#     }
#     artists_df = pd.DataFrame(data)
    
#     # Initialize and prepare recommender
#     recommender = ArtistRecommender()
#     recommender.prepare_artist_data(artists_df)
    
#     # Get recommendations
#     campaign_req = """Looking for pop artists with strong social media presence 
#                      for a youth-focused summer campaign in North America"""
    
#     recommendations = recommender.get_recommendations(campaign_req)
    
#     # Apply additional filters
#     filtered_recommendations = recommender.filter_recommendations(
#         recommendations,
#         min_subscribers=500000,
#         countries=['USA', 'Canada'],
#         genres=['Pop']
#     )