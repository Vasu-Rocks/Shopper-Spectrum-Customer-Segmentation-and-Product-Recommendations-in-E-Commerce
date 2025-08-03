import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Load models and data ---
scaler = joblib.load('scaler_rfm.pkl')
kmeans = joblib.load('kmeans_rfm.pkl')

# Load cleaned data (just for item matrix; replace with your actual cleaned CSV if needed)
df_cleaned = pd.read_csv('C:/Users/gurus/OneDrive/Documents/GUVI_PROJECT/New folder/online_retail.csv')

# Pre-process data for recommendation
basket = df_cleaned.pivot_table(index='CustomerID', columns='Description', values='Quantity', fill_value=0)
item_sim_matrix = cosine_similarity(basket.T)
item_sim_df = pd.DataFrame(item_sim_matrix, index=basket.columns, columns=basket.columns)

# --- Streamlit UI ---
st.title("🛍️ E-commerce Customer & Product Intelligence")

# Tabs
tab1, tab2 = st.tabs(["🔄 Product Recommendation", "👤 Customer Segmentation"])

# --------------------------------------
# 🎯 1️⃣ Product Recommendation Module
# --------------------------------------
with tab1:
    st.header("Product Recommendation System")
    product_name = st.text_input("Enter a product name:")
    if st.button("Get Recommendations"):
        if product_name not in item_sim_df.columns:
            st.error("Product not found. Please check spelling or try another product.")
        else:
            similar_items = item_sim_df[product_name].sort_values(ascending=False).iloc[1:6]
            st.success(f"Top 5 recommendations for **{product_name}**:")
            for i, (item, score) in enumerate(similar_items.items(), start=1):
                st.markdown(f"**{i}. {item}** (Similarity Score: {score:.2f})")

# --------------------------------------
# 🎯 2️⃣ Customer Segmentation Module
# --------------------------------------
with tab2:
    st.header("Customer Segmentation Prediction")
    recency = st.number_input("Recency (days since last purchase):", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases):", min_value=1, value=5)
    monetary = st.number_input("Monetary (total spend):", min_value=1.0, value=500.0, step=10.0)

    if st.button("Predict Cluster"):
        rfm_input = pd.DataFrame({
            'Recency': [recency],
            'Frequency': [frequency],
            'Monetary': [monetary]
        })

        rfm_scaled_input = scaler.transform(rfm_input)
        cluster_label = kmeans.predict(rfm_scaled_input)[0]

        # Segment mapping logic (customize if needed)
        if recency <= 30 and frequency >= 10 and monetary >= 1000:
            segment = 'High-Value'
        elif frequency >= 5 and monetary >= 500:
            segment = 'Regular'
        elif recency >= 90:
            segment = 'At-Risk'
        else:
            segment = 'Occasional'

        st.success(f"Predicted Cluster: **{cluster_label}** — Segment: **{segment}**")

