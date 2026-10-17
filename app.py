# =======NEWS RECOMMENDATION SYSTEM=========
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Live News Recommendation System", layout="wide")
st.title("📰 Live News Recommendation & Reader")

# Enter your NewsAPI key here
API_KEY = "c73d46f022404d57947189d9e4814c30"  

@st.cache_data(ttl=3600)
def fetch_live_news():
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    articles_list = []
    if data.get("status") == "ok":
        for article in data.get("articles", []):
            if article.get("title") and article.get("description"):
                articles_list.append({
                    "Title": article.get("title"),
                    "Description": article.get("description"),
                    "Category": article.get("source", {}).get("name", "General"),
                    "Image": article.get("urlToImage"),
                    "URL": article.get("url")
                })
    return pd.DataFrame(articles_list)

try:
    data = fetch_live_news()

    if data.empty:
        st.warning("News load nahi ho pa rahi hai. Kripya API Key check karein.")
    else:
        # Build Recommendation Model
        @st.cache_resource
        def build_model(df):
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(df["Title"].fillna("") + " " + df["Description"].fillna(""))
            return cosine_similarity(tfidf_matrix)

        similarity = build_model(data)
        indices = pd.Series(data.index, index=data["Title"]).drop_duplicates()

        # UI Dropdown
        st.subheader("🔍 News Select Karein:")
        selected_title = st.selectbox("Choose an article:", options=data["Title"].values)

        # Selected Article Display
        selected_article = data[data["Title"] == selected_title].iloc[0]
        st.markdown("---")
        st.subheader(f"📌 {selected_article['Title']}")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if selected_article["Image"]:
                st.image(selected_article["Image"], use_container_width=True)
            else:
                st.info("No image available")
        with col2:
            st.write(selected_article["Description"])
            st.markdown(f"[👉 Read Full Article Here]({selected_article['URL']})")

        # Recommendations
        st.markdown("---")
        st.subheader("💡 Related News Recommendations:")
        
        idx = indices[selected_title]
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
        
        rec_cols = st.columns(3)
        for i, (rec_idx, score) in enumerate(sim_scores):
            rec_item = data.iloc[rec_idx]
            with rec_cols[i]:
                if rec_item["Image"]:
                    st.image(rec_item["Image"], use_container_width=True)
                st.bold(rec_item["Title"])
                st.caption(rec_item["Description"][:100] + "...")
                st.markdown(f"[Read Article]({rec_item['URL']})")

except Exception as e:
    st.error(f"Error: {e}")
            st.write(f"**{i}.** {news}")

except Exception as e:
    st.error(f"Error loading application: {e}")
