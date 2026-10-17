# =======NEWS RECOMMENDATION SYSTEM=========

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="News Recommendation System", layout="wide")
st.title("📰 News Recommendation System")

# Step 1: Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("News.csv")

try:
    data = load_data()

    # Step 2: Show Category Distribution (Plotly Fix)
    st.subheader("📊 News Categories Distribution")
    category_counts = data["News Category"].value_counts().reset_index()
    category_counts.columns = ["News Category", "Count"]

    figure = px.bar(
        category_counts, 
        x="News Category", 
        y="Count", 
        title="Types of News Categories",
        color="News Category"
    )
    st.plotly_chart(figure, use_container_width=True)

    # Step 3: Recommendation Engine
    @st.cache_resource
    def build_model(df):
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df["Title"].fillna(""))
        return cosine_similarity(tfidf_matrix)

    similarity = build_model(data)
    indices = pd.Series(data.index, index=data["Title"]).drop_duplicates()

    # Step 4: Recommendation Logic
    def news_recommendation(title):
        if title not in indices:
            return []
        idx = indices[title]
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        news_indices = [i[0] for i in sim_scores]
        return data["Title"].iloc[news_indices]

    # Step 5: UI Dropdown & Output
    st.subheader("🔍 Get News Recommendations")
    selected_news = st.selectbox("Select a News Title:", options=data["Title"].values)

    if st.button("Recommend"):
        recommendations = news_recommendation(selected_news)
        st.write("### Recommended Articles:")
        for i, news in enumerate(recommendations, 1):
            st.write(f"**{i}.** {news}")

except Exception as e:
    st.error(f"Error loading application: {e}")
