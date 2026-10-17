# =======NEWS RECOMMENDATION SYSTEM=========
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="News Recommendation System", layout="wide")
st.title("📰 News Recommendation System & Reader")

# Optional: Add NewsAPI key if running locally or using a paid key
API_KEY = "c73d46f022404d57947189d9e4814c30"  

@st.cache_data(ttl=3600)
def fetch_news():
    # Try fetching live news via API first
    if API_KEY and API_KEY != "YOUR_NEWSAPI_KEY_HERE":
        try:
            url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEY}"
            response = requests.get(url, timeout=5)
            data = response.json()
            if data.get("status") == "ok" and data.get("articles"):
                articles = []
                for a in data.get("articles", []):
                    if a.get("title"):
                        articles.append({
                            "Title": a.get("title"),
                            "Description": a.get("description") or "Click full article link below to read details.",
                            "Category": a.get("source", {}).get("name", "General"),
                            "Image": a.get("urlToImage"),
                            "URL": a.get("url") or "#"
                        })
                df = pd.DataFrame(articles)
                if not df.empty:
                    return df
        except Exception:
            pass

    # Fallback to local CSV dataset with image/link defaults if API fails or fails on Cloud
    df_csv = pd.read_csv("News.csv")
    if "Description" not in df_csv.columns:
        df_csv["Description"] = df_csv["Title"] + " - Read detailed coverage on this top story."
    if "Image" not in df_csv.columns:
        df_csv["Image"] = "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=500"
    if "URL" not in df_csv.columns:
        df_csv["URL"] = "https://news.google.com"
    return df_csv

try:
    data = fetch_news()

    # Build Recommendation Engine
    @st.cache_resource
    def build_model(df):
        tfidf = TfidfVectorizer(stop_words="english")
        combined_text = df["Title"].fillna("") + " " + df["Description"].fillna("")
        tfidf_matrix = tfidf.fit_transform(combined_text)
        return cosine_similarity(tfidf_matrix)

    similarity = build_model(data)
    indices = pd.Series(data.index, index=data["Title"]).drop_duplicates()

    # News Selector
    st.subheader("🔍 Select a News Article:")
    selected_title = st.selectbox("Choose Headline:", options=data["Title"].values)

    # Main Article View
    selected_article = data[data["Title"] == selected_title].iloc[0]
    st.markdown("---")
    st.subheader(f"📌 {selected_article['Title']}")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        img_url = selected_article["Image"] if pd.notna(selected_article["Image"]) else "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=500"
        st.image(img_url, use_container_width=True)
    with col2:
        st.write(selected_article["Description"])
        st.markdown(f"[👉 **Read Full Article Here**]({selected_article['URL']})")

    # Recommendations
    st.markdown("---")
    st.subheader("💡 Related Recommendations:")
    
    idx = indices[selected_title]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
    
    rec_cols = st.columns(3)
    for i, (rec_idx, score) in enumerate(sim_scores):
        rec_item = data.iloc[rec_idx]
        with rec_cols[i]:
            rec_img = rec_item["Image"] if pd.notna(rec_item["Image"]) else "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=500"
            st.image(rec_img, use_container_width=True)
            st.markdown(f"**{rec_item['Title']}**")
            st.caption(f"{str(rec_item['Description'])[:90]}...")
            st.markdown(f"[Read Article]({rec_item['URL']})")

except Exception as e:
    st.error(f"Error loading app: {e}")
