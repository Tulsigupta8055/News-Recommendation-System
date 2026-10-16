# =======NEWS RECOMMENDATION SYSTEM=========

# Step 1:Import Necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Step 2:Load the Dataset

data=pd.read_csv("News.csv")
print(data.head())

# Step 3:Visualizing News Categories

categories = data["News Category"].value_counts()
label = categories.index
counts = categories.values

figure = px.bar(data, x=label, y=counts, title="Types of News Categores")
figure.show()

#Step 4: Convert News Titles into Numerical Vectors

feature = data["Title"].astype(str).tolist()
tfidf = TfidfVectorizer(input='content', stop_words="english") #changed input to 'content'
tfidf_matrix=tfidf.fit_transform(feature)

# Step 5: Calculate cosine Similarity

similarity = cosine_similarity(tfidf_matrix)

# Step 6: Create a Function for News Recommendation

indices = pd.Series(data.index, index = data["Title"]).drop_duplicates()

def news_recommendation(title, similarity=similarity):
    index = indices[title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11] # Exclude the input news itself
    news_indices = [i[0] for i in similarity_scores]
    return data ["Title"].iloc[news_indices]

# Step 7: Test the News Recommendation System

print(news_recommendation("Walmart slashes prices on Last-Generation ipads"))





