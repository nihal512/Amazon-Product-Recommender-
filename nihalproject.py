import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("amazon.csv")
    df['combined_features'] = (
        df['product_name'].fillna('') + ' ' +
        df['about_product'].fillna('') + ' ' +
        df['category'].fillna('')
    )
    return df

# Compute TF-IDF and similarity matrix
@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    product_name_to_indices = defaultdict(list)
    for idx, name in enumerate(df['product_name']):
        product_name_to_indices[name].append(idx)

    return cosine_sim, product_name_to_indices

# Recommendation function
def get_recommendations(product_name, df, cosine_sim, product_name_to_indices, top_n=5):
    indices = product_name_to_indices.get(product_name)
    if not indices:
        return []

    idx = indices[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [score for score in sim_scores if score[0] != idx]
    sim_scores = sim_scores[:top_n]

    recommended_indices = [i[0] for i in sim_scores]
    similarities = [score[1] for score in sim_scores]
    avg_similarity = sum(similarities) / len(similarities)

    return df[['product_name', 'product_link']].iloc[recommended_indices], avg_similarity

# Streamlit UI
st.set_page_config(page_title="E-Commerce Product Recommender", layout="wide")
st.title("🛍 Amazon Product Recommender")

df = load_data()
cosine_sim, product_name_to_indices = compute_similarity(df)

# Search bar + dropdown
product_list = sorted(df['product_name'].unique())
search_query = st.text_input("🔍 Search for a product:")
filtered_products = [p for p in product_list if search_query.lower() in p.lower()] if search_query else product_list

if search_query and not filtered_products:
    st.warning("No products found matching your search.")

selected_product = st.selectbox("Choose a product to get recommendations:", filtered_products)

if st.button("Get Recommendations"):
    recommendations, avg_similarity = get_recommendations(selected_product, df, cosine_sim, product_name_to_indices)

    if len(recommendations) == 0:
        st.warning("No recommendations found for this product.")
    else:
        st.subheader("🔎 Recommended Products:")
        st.markdown(f"*Similarity Score:* {avg_similarity * 100:.2f}%")
        for _, row in recommendations.iterrows():
            st.markdown(f"- [{row['product_name']}]({row['product_link']})")