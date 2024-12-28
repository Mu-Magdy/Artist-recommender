import streamlit as st
import pandas as pd
from recommender import ArtistRecommender
import plotly.express as px
from typing import List, Dict

def load_sample_data() -> pd.DataFrame:
    """Load sample artist data"""
    data_path = 'artist_dataset.csv'
    return pd.read_csv(data_path)

def display_recommendations(recommendations: List[Dict]):
    """Display recommendations in a formatted way"""
    if not recommendations:
        st.warning("No artists found matching your criteria.")
        return

    # Create a DataFrame for easier visualization
    df = pd.DataFrame(recommendations)
    
    # Display results in a table
    st.subheader("Recommended Artists")
    st.dataframe(
        df.style.format({
            'similarity_score': '{:.2%}',
            'subscribers': '{:,.0f}'
        })
    )
    
    # Create visualizations
    fig = px.bar(df, 
                 x='artist', 
                 y='subscribers',
                 color='genre',
                 title='Subscriber Count by Artist')
    st.plotly_chart(fig)

    # Create a pie chart for genre distribution
    genre_dist = df['genre'].value_counts()
    fig_pie = px.pie(values=genre_dist.values,
                     names=genre_dist.index,
                     title='Genre Distribution')
    st.plotly_chart(fig_pie)

def main():
    st.set_page_config(
        page_title="Artist Recommendation System",
        page_icon="🎵",
        layout="wide"
    )

    st.title("🎵 Marketing Campaign Artist Recommender")
    st.markdown("""
    Find the perfect artists for your marketing campaign using AI-powered recommendations.
    """)

    # Initialize recommender
    try:
        recommender = ArtistRecommender()
    except Exception as e:
        st.error(f"Error initializing recommender system: {str(e)}")
        return

    # Load and prepare data
    with st.spinner("Loading artist data..."):
        df = load_sample_data()
        recommender.prepare_artist_data(df)

    # Sidebar filters
    st.sidebar.header("Campaign Requirements")
    
    # Campaign description
    campaign_description = st.sidebar.text_area(
        "Describe your campaign requirements",
        placeholder="e.g., Looking for pop artists with strong social media presence for a youth-focused summer campaign"
    )

    # Filters
    st.sidebar.subheader("Additional Filters")
    
    min_subscribers = st.sidebar.number_input(
        "Minimum Subscribers",
        min_value=0,
        max_value=100000000,
        value=10000,
        step=10000
    )

    available_countries = df['country'].unique().tolist()
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=available_countries,
        # default=available_countries
    )

    available_genres = df['genre'].unique().tolist()
    selected_genres = st.sidebar.multiselect(
        "Select Genres",
        options=available_genres,
        # default=available_genres
    )

    # Number of recommendations
    top_k = st.sidebar.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=50,
        value=5
    )

    # Get recommendations button
    if st.sidebar.button("Get Recommendations"):
        if not campaign_description:
            st.warning("Please enter campaign requirements.")
            return

        with st.spinner("Finding the best artists for your campaign..."):
            try:
                # Get initial recommendations
                recommendations = recommender.get_recommendations(
                    campaign_description,
                    top_k=top_k
                )
                
                # Apply filters
                filtered_recommendations = recommender.filter_recommendations(
                    recommendations,
                    min_subscribers=min_subscribers,
                    countries=selected_countries,
                    genres=selected_genres
                )
                
                # Display results
                display_recommendations(filtered_recommendations)
                
            except Exception as e:
                st.error(f"Error getting recommendations: {str(e)}")

    # Display current artist database
    st.subheader("Current Artist Database")
    st.dataframe(df)

if __name__ == "__main__":
    main()