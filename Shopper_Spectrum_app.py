import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import io

# ------------------ Page Configuration ------------------
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# ------------------ Load Models and Data ------------------
@st.cache_resource
def load_models():
    try:
        with open('pipeline.pkl', 'rb') as f:
            pipeline = joblib.load(f)
        with open('cluster_labels.pkl', 'rb') as f:
            cluster_labels = joblib.load(f)
    except FileNotFoundError:
        pipeline, cluster_labels = None, None

    try:
        with open('dashboard_data.pkl', 'rb') as f:
            dashboard_data = joblib.load(f)
    except FileNotFoundError:
        dashboard_data = None

    try:
        with open('similarity_matrix.pkl', 'rb') as f:
            similarity_matrix = joblib.load(f)
        with open('product_map.pkl', 'rb') as f:
            product_map = joblib.load(f)
    except FileNotFoundError:
        similarity_matrix, product_map = None, None
        
    try:
        with open('customer_profiles.pkl', 'rb') as f:
            customer_profiles = joblib.load(f)
        with open('customer_history.pkl', 'rb') as f:
            customer_history = joblib.load(f)
    except FileNotFoundError:
        customer_profiles, customer_history = None, None

    try:
        with open('association_rules.pkl', 'rb') as f:
            association_rules = joblib.load(f)
    except FileNotFoundError:
        association_rules = None

    return pipeline, cluster_labels, dashboard_data, similarity_matrix, product_map, customer_profiles, customer_history, association_rules

pipeline, cluster_labels, dashboard_data, similarity_matrix, product_map, customer_profiles, customer_history, association_rules = load_models()

st.sidebar.title("Shopper Spectrum")
page = st.sidebar.radio("Select Module", ["Dashboard", "Customer Profile", "Clustering", "Recommendation"])

# ------------------ Dashboard ------------------
if page == "Dashboard":
    st.title("Store Performance Dashboard")
    
    if dashboard_data is None:
        st.error("Dashboard data is missing. Please run `generate_models.py` to generate it.")
    else:
        st.write("Welcome to the interactive Shopper Spectrum Dashboard. Explore top-level store metrics below.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Selling Products")
            top_products = dashboard_data['top_products']
            df_top = pd.DataFrame(list(top_products.items()), columns=['Product', 'Quantity Sold'])
            fig_top = px.bar(df_top, x='Quantity Sold', y='Product', orientation='h', color='Quantity Sold', color_continuous_scale='Blues')
            fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)
            
        with col2:
            st.subheader("Top 10 Countries by Transactions")
            top_countries = dashboard_data['transactions_by_country']
            df_country = pd.DataFrame(list(top_countries.items()), columns=['Country', 'Transactions'])
            fig_country = px.bar(df_country, x='Country', y='Transactions', color='Transactions', color_continuous_scale='Teal')
            st.plotly_chart(fig_country, use_container_width=True)

# ------------------ Customer Profile ------------------
elif page == "Customer Profile":
    st.title("Customer Profile")
    st.write("Search for an existing customer to view their complete profile, purchase history, and personalized recommendations.")
    
    if customer_profiles is None or customer_history is None:
        st.error("Customer data missing. Please run `generate_models.py`.")
    else:
        cust_id_input = st.text_input("Enter Customer ID (e.g., 17850.0)")
        
        if cust_id_input:
            try:
                cust_id = float(cust_id_input)
                profile = customer_profiles[customer_profiles['CustomerID'] == cust_id]
                
                if profile.empty:
                    st.warning("Customer ID not found in database.")
                else:
                    st.markdown("---")
                    st.header(f"Profile: {cust_id}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Recency", f"{int(profile['Recency'].values[0])} days")
                    c2.metric("Frequency", f"{int(profile['Frequency'].values[0])} orders")
                    c3.metric("Monetary", f"${profile['Monetary'].values[0]:.2f}")
                    
                    segment = profile['Segment'].values[0]
                    color = "green" if segment == "High-Value" else "orange" if segment == "Regular" else "red" if segment == "At-Risk" else "blue"
                    c4.markdown(f"### Segment\n<span style='color:{color}; font-weight:bold;'>{segment}</span>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    col_hist, col_rec = st.columns(2)
                    
                    history_codes = customer_history.get(cust_id, [])
                    history_names = []
                    
                    with col_hist:
                        st.subheader("Past Purchases")
                        if not history_codes:
                            st.write("No item history found.")
                        else:
                            for code in history_codes:
                                if code in product_map.index:
                                    desc = product_map.loc[code]['Description']
                                    if isinstance(desc, pd.Series): desc = desc.iloc[0]
                                    history_names.append(desc)
                                    st.markdown(f"- {desc}")
                                    
                    with col_rec:
                        st.subheader("Personalized Recommendations")
                        if not history_codes or association_rules is None or association_rules.empty:
                            st.write("Not enough data for personalized recommendations.")
                        else:
                            st.write("*Based on Market Basket Analysis of their past purchases:*")
                            recommended_items = set()
                            for code in history_codes:
                                relevant_rules = association_rules[association_rules['antecedents'].apply(lambda x: code in x)]
                                top_rules = relevant_rules.sort_values(by='lift', ascending=False).head(3)
                                for _, row in top_rules.iterrows():
                                    for cons in row['consequents']:
                                        if cons not in history_codes:
                                            recommended_items.add(cons)
                            
                            if not recommended_items:
                                st.write("No strong new recommendations found.")
                            else:
                                for rec_code in list(recommended_items)[:5]:
                                    if rec_code in product_map.index:
                                        desc = product_map.loc[rec_code]['Description']
                                        if isinstance(desc, pd.Series): desc = desc.iloc[0]
                                        st.markdown(f"- **{desc}**")
            except ValueError:
                st.error("Please enter a valid numeric Customer ID.")

# ------------------ Clustering ------------------
elif page == "Clustering":
    st.title("Customer Segmentation")
    
    if pipeline is None or cluster_labels is None:
        st.error("Model files are missing. Please run `generate_models.py`.")
    else:
        tab1, tab2 = st.tabs(["Individual Prediction", "Batch Prediction"])
        
        with tab1:
            st.subheader("Predict Segment for a Single Customer")
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
            frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5)
            monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

            if st.button("Predict Segment"):
                input_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
                cluster_id = pipeline.predict(input_df)[0]
                segment = cluster_labels.get(cluster_id, "Unknown")
                
                if segment == "High-Value":
                    st.success(f"This customer belongs to: **{segment}**")
                elif segment == "At-Risk":
                    st.error(f"This customer belongs to: **{segment}**")
                else:
                    st.info(f"This customer belongs to: **{segment}**")

        with tab2:
            st.subheader("Batch Prediction from CSV")
            st.write("Upload a CSV file containing `Recency`, `Frequency`, and `Monetary` columns.")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df_batch = pd.read_csv(uploaded_file)
                    req_cols = ['Recency', 'Frequency', 'Monetary']
                    if not all(col in df_batch.columns for col in req_cols):
                        st.error(f"CSV must contain the following columns: {req_cols}")
                    else:
                        st.write("Preview of uploaded data:")
                        st.dataframe(df_batch.head())
                        
                        if st.button("Run Batch Prediction"):
                            with st.spinner('Predicting segments...'):
                                features = df_batch[req_cols]
                                predictions = pipeline.predict(features)
                                df_batch['Segment'] = [cluster_labels.get(p, "Unknown") for p in predictions]
                                
                                st.success("Batch prediction complete!")
                                st.write("Results preview:")
                                st.dataframe(df_batch.head(10))
                                
                                csv = df_batch.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=csv,
                                    file_name='segmented_customers.csv',
                                    mime='text/csv',
                                )
                except Exception as e:
                    st.error(f"Error processing file: {e}")

# ------------------ Recommendation ------------------
elif page == "Recommendation":
    st.title("Product Recommender")

    if similarity_matrix is None or product_map is None:
        st.error("Recommendation model files are missing. Please run `generate_models.py`.")
    else:
        st.write("Select a product to find recommendations based on customer purchase history.")
        
        unique_products = product_map['Description'].dropna().unique().tolist()
        unique_products.sort()
        
        selected_product = st.selectbox("Search and Select a Product", unique_products)

        if st.button("Recommend"):
            matches = product_map[product_map['Description'] == selected_product]
            if matches.empty:
                st.error("Product not found.")
            else:
                stock_code = matches.index[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"### Frequently Bought Together")
                    st.caption("Based on Market Basket Analysis (items bought in the exact same cart).")
                    
                    if association_rules is None or association_rules.empty:
                        st.info("No Market Basket rules generated yet.")
                    else:
                        relevant_rules = association_rules[association_rules['antecedents'].apply(lambda x: stock_code in x)]
                        if relevant_rules.empty:
                            st.write("No strong association rules found for this specific item.")
                        else:
                            top_rules = relevant_rules.sort_values(by=['lift', 'confidence'], ascending=False).head(5)
                            for _, row in top_rules.iterrows():
                                for cons in row['consequents']:
                                    if cons in product_map.index:
                                        c_desc = product_map.loc[cons]['Description']
                                        if isinstance(c_desc, pd.Series): c_desc = c_desc.iloc[0]
                                        st.markdown(f"- **{c_desc}** *(Confidence: {row['confidence']:.0%})*")

                with col2:
                    st.write(f"### Customers Also Liked")
                    st.caption("Based on overall purchase patterns (Cosine Similarity).")
                    if stock_code not in similarity_matrix.index:
                        st.warning("Product found, but no similarity data available.")
                    else:
                        similar_items = similarity_matrix[stock_code].sort_values(ascending=False).drop(stock_code).head(5)
                        for code in similar_items.index:
                            if code in product_map.index:
                                desc = product_map.loc[code]['Description']
                                if isinstance(desc, pd.Series): desc = desc.iloc[0]
                                score = similar_items[code]
                                if score > 0:
                                    st.markdown(f"- **{desc}** *(Similarity: {score:.2f})*")
