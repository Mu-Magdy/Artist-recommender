# Artist Recommender System for Marketing Campaigns 🎵

A powerful recommendation system that helps match artists with marketing campaigns using AI-powered semantic search and FAISS vector similarity. The system includes both a core recommendation engine and a user-friendly Streamlit interface.

## 🌟 Features

- **Semantic Search**: Uses state-of-the-art language models to understand campaign requirements
- **Vector Similarity**: Employs FAISS for efficient similarity search
- **Interactive UI**: Built with Streamlit for easy campaign matching
- **Data Visualization**: Includes dynamic charts and graphs for better insights
- **Flexible Filtering**: Filter by subscribers, country, genre, and more
- **Local Processing**: All processing happens locally - no external APIs needed

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Mu-Magdy/Artist-recommender.git
cd artist-recommender
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```



## 🚀 Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Access the web interface at `http://localhost:8501`

3. Use the application:
   - Enter your campaign requirements in the text area
   - Adjust filters using the sidebar controls
   - Click "Get Recommendations" to see matching artists
   - View visualizations and detailed metrics

## 🔧 Configuration

The system uses several configurable parameters:

- `index_path`: Location for storing FAISS index (default: 'artist_index')
- `dimension`: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
- `top_k`: Number of recommendations to return (configurable in UI)

## 🎨 Customization

### Adding New Features

1. To add new artist features:
   - Update the `prepare_artist_data` method in `ArtistRecommender`
   - Modify the data preprocessing pipeline
   - Update the Streamlit UI to include new filters

2. To modify the recommendation algorithm:
   - Adjust the FAISS index type in `ArtistRecommender.__init__`
   - Modify the similarity calculation in `get_recommendations`

## 📊 Data Format

The system expects artist data in a pandas DataFrame with the following columns:
- `name`: Artist name (string)
- `country`: Country of origin (string)
- `subscribers`: Number of subscribers/followers (integer)
- `genre`: Music genre (string)
