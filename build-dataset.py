import pandas as pd
import random

# Step 1: Define the Dataset Structure
columns = ["Artist_ID", "Name", "Genre", "Subscribers", "Engagement_Score", "Region"]

data = []

# Step 2: Generate Synthetic Data

# Define some genres and regions for variety
genres = ["Pop", "Rock", "Jazz", "Classical", "Hip-Hop", "Country", "Electronic"]
regions = ["North America", "Europe", "Asia", "South America", "Africa", "Australia"]

# Generate 1000 artists with random attributes
for artist_id in range(1, 1001):
    name = f"Artist_{artist_id}"
    genre = random.choice(genres)
    subscribers = random.randint(1000, 1000000)  # Between 1k and 1M
    engagement_score = round(random.uniform(0.1, 10.0), 2)  # Between 0.1 and 10.0
    region = random.choice(regions)
    
    data.append([artist_id, name, genre, subscribers, engagement_score, region])

# Step 3: Create DataFrame
dataset = pd.DataFrame(data, columns=columns)

# Step 4: Save to CSV
output_file = "artist_dataset.csv"
dataset.to_csv(output_file, index=False)

print(f"Dataset created and saved to {output_file}")
