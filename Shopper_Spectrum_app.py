import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------ Page Configuration ------------------
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# ------------------ Load Models and Data ------------------
@st.cache_resource
def load_models():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('similarity_matrix.pkl', 'rb') as f:
        similarity_matrix = pickle.load(f)
    with open('product_map.pkl', 'rb') as f:
        product_map = pickle.load(f)
    return kmeans, scaler, similarity_matrix, product_map

kmeans, scaler, similarity_matrix, product_map = load_models()


st.sidebar.title("🛍️ Shopper Spectrum")
page = st.sidebar.radio("Select Module", ["Home", "Clustering", "Recommendation"])

# ------------------ Home ------------------
if page == "Home":
    st.title("📊 Shopper Spectrum Dashboard")
    st.write("""
        Welcome to the Shopper Spectrum App!
        - 🎯 Predict customer segments using RFM values.
        - 🎯 Get product recommendations using collaborative filtering.
    """)

# ------------------ Clustering ------------------
elif page == "Clustering":
    st.title("🧠 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

    if st.button("Predict Segment"):
        scaled_input = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(scaled_input)[0]

        # Define labels manually
        cluster_labels = {
            0: "Occasional Shopper",
            1: "At-Risk",
            2: "High-Value",
            3: "Regular"
        }
        segment = cluster_labels.get(cluster, "Unknown")
        st.success(f"This customer belongs to: **{segment}**")

# ------------------ Recommendation ------------------
elif page == "Recommendation":
    st.title("🔁 Product Recommender")

    product_name = st.text_input("Enter Product Name")

    if st.button("Recommend"):
        # Reverse lookup from Description to StockCode
        matches = product_map[product_map['Description'].str.upper() == product_name.upper()]
        if matches.empty:
            st.error("❌ Product not found.")
        else:
            stock_code = matches.index[0]
            if stock_code not in similarity_matrix.index:
                st.warning("Product found, but no recommendations available.")
            else:
                similar_items = similarity_matrix[stock_code].sort_values(ascending=False)[1:6]
                st.write("### Recommended Products:")
                for code in similar_items.index:
                    desc = product_map.loc[code]['Description']
                    st.markdown(f"- **{desc}**")

