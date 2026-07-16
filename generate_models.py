import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import joblib
import os
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

def load_and_clean_data(filepath="OnlineRetail.csv"):
    print("Loading data...")
    df = pd.read_csv(filepath, encoding='latin1')
    print(f"Original shape: {df.shape}")
    
    # 1. Drop missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    
    # 2. Remove cancelled invoices
    cancelled_invoices = df[df["InvoiceNo"].astype(str).str.startswith('C')]
    df = df.drop(index=cancelled_invoices.index)
    
    # 3. Remove negative/zero quantity and price
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # 4. Drop duplicates
    df = df.drop_duplicates()
    
    # Add Revenue
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    print(f"Cleaned shape: {df.shape}")
    return df

def generate_dashboard_data(df):
    print("Generating dashboard data...")
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).to_dict()
    transactions_by_country = df.groupby('Country')['InvoiceNo'].nunique().sort_values(ascending=False).head(10).to_dict()
    
    dashboard_data = {
        'top_products': top_products,
        'transactions_by_country': transactions_by_country
    }
    
    with open('dashboard_data.pkl', 'wb') as f:
        joblib.dump(dashboard_data, f)

def create_rfm_features(df):
    print("Creating RFM features...")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Revenue': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

def train_segmentation_model(rfm):
    print("Training segmentation model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=4, random_state=42, n_init=10))
    ])
    
    features = rfm[['Recency', 'Frequency', 'Monetary']]
    pipeline.fit(features)
    
    rfm['Cluster'] = pipeline.predict(features)
    cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    sorted_clusters = cluster_summary.sort_values(by='Monetary').index.tolist()
    robust_labels = {
        sorted_clusters[0]: "At-Risk",
        sorted_clusters[1]: "Occasional Shopper",
        sorted_clusters[2]: "Regular",
        sorted_clusters[3]: "High-Value",
    }
    
    with open('pipeline.pkl', 'wb') as f:
        joblib.dump(pipeline, f)
        
    with open('cluster_labels.pkl', 'wb') as f:
        joblib.dump(robust_labels, f)
        
    return pipeline, robust_labels, rfm

def generate_customer_profiles(rfm, robust_labels, df):
    print("Generating customer profiles and history...")
    # Add Segment text label
    rfm['Segment'] = rfm['Cluster'].map(robust_labels)
    
    with open('customer_profiles.pkl', 'wb') as f:
        joblib.dump(rfm, f)
        
    # Get top 10 unique items purchased by each customer
    print("Generating customer history...")
    # Group by CustomerID and StockCode, count frequency
    cust_item_counts = df.groupby(['CustomerID', 'StockCode']).size().reset_index(name='count')
    # Sort and take top 10 per customer
    top_items_per_cust = cust_item_counts.sort_values(['CustomerID', 'count'], ascending=[True, False]).groupby('CustomerID').head(10)
    
    customer_history = top_items_per_cust.groupby('CustomerID')['StockCode'].apply(list).to_dict()
    
    with open('customer_history.pkl', 'wb') as f:
        joblib.dump(customer_history, f)

def build_recommender(df):
    print("Building similarity matrix (sparse)...")
    product_map = df[['StockCode', 'Description']].drop_duplicates(subset='StockCode').dropna().set_index('StockCode')
    with open('product_map.pkl', 'wb') as f:
        joblib.dump(product_map, f)
    
    customers = df['CustomerID'].astype('category')
    items = df['StockCode'].astype('category')
    
    customer_cat = customers.cat.categories
    item_cat = items.cat.categories
    
    item_customer_sparse = sp.coo_matrix(
        (df['Quantity'], (items.cat.codes, customers.cat.codes)),
        shape=(len(item_cat), len(customer_cat))
    ).tocsr()
    
    cosine_sim_matrix = cosine_similarity(item_customer_sparse)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=item_cat, columns=item_cat)
    
    with open('similarity_matrix.pkl', 'wb') as f:
        joblib.dump(cosine_sim_df, f)

def build_market_basket(df):
    print("Building market basket analysis (top 500 items)...")
    # To prevent memory errors, only consider the top 500 selling items
    top_items = df.groupby('StockCode')['Quantity'].sum().nlargest(500).index
    df_top = df[df['StockCode'].isin(top_items)]
    
    # Create basket
    basket = (df_top.groupby(['InvoiceNo', 'StockCode'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
              
    # Convert to boolean
    basket = basket.map(lambda x: True if x > 0 else False)
    
    print("Running FP-Growth...")
    frequent_itemsets = fpgrowth(basket, min_support=0.02, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("Warning: No frequent itemsets found with current support threshold.")
        rules = pd.DataFrame()
    else:
        print("Generating Association Rules...")
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    
    with open('association_rules.pkl', 'wb') as f:
        joblib.dump(rules, f)
    print("Saved association_rules.pkl")

def main():
    if not os.path.exists("OnlineRetail.csv"):
        print("Error: OnlineRetail.csv not found.")
        return
        
    df = load_and_clean_data("OnlineRetail.csv")
    generate_dashboard_data(df)
    
    rfm = create_rfm_features(df)
    pipeline, robust_labels, rfm = train_segmentation_model(rfm)
    
    generate_customer_profiles(rfm, robust_labels, df)
    
    build_recommender(df)
    build_market_basket(df)
    
    print("Model generation complete.")

if __name__ == "__main__":
    main()
