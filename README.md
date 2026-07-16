# 🛍️ Shopper Spectrum: E-Commerce Customer CRM & Recommender

Shopper Spectrum is an end-to-end Machine Learning web application designed to help e-commerce businesses understand their customers better and increase sales through personalized recommendations. 

Built with **Streamlit**, **Scikit-Learn**, and **mlxtend**, this project processes raw transaction data to build a robust Customer Relationship Management (CRM) dashboard.

---

## 🌟 Key Features

*   **📊 Store Performance Dashboard**: An interactive overview of the store's top-performing metrics, including best-selling products and transaction volumes by country, visualized with Plotly.
*   **👤 Customer Profile (CRM)**: Look up any customer by their ID to get a 360-degree view of their purchasing behavior. View their RFM (Recency, Frequency, Monetary) stats, their assigned value segment, their complete purchase history, and receive **personalized recommendations** on what they are most likely to buy next.
*   **🧠 Customer Segmentation (RFM + KMeans)**: Uses K-Means clustering to automatically group customers into segments (`High-Value`, `Regular`, `Occasional Shopper`, `At-Risk`). 
    *   *Individual Prediction:* Enter manual RFM values to instantly predict a customer's segment.
    *   *Batch Prediction:* Upload a CSV of RFM values to segment thousands of customers simultaneously and download the results.
*   **🔁 Advanced Product Recommender**: 
    *   *Frequently Bought Together*: Uses Market Basket Analysis (**FP-Growth algorithm**) to find strong association rules between items in the exact same cart.
    *   *Customers Also Liked*: Uses **Cosine Similarity** (Collaborative Filtering) to recommend products based on broader customer purchase patterns.

---

## 🛠️ Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (KMeans, StandardScaler, Cosine Similarity), [mlxtend](http://rasbt.github.io/mlxtend/) (FP-Growth, Association Rules)
*   **Data Manipulation**: Pandas, NumPy, SciPy (Sparse Matrices for memory efficiency)
*   **Visualization**: Plotly

---

## 📂 Project Architecture

The project is split into two main components to separate heavy model training from the lightweight web application:

1.  `generate_models.py`: The data engineering and machine learning pipeline. It cleans the raw data, engineers RFM features, trains the KMeans pipeline, runs the FP-Growth algorithm, computes sparse similarity matrices, and exports all necessary `.pkl` files.
2.  `Shopper_Spectrum_app.py`: The Streamlit web interface that loads the exported models to provide real-time dashboards and predictions without needing to process the massive raw dataset on the fly.

---

## 🚀 Installation & Setup

Follow these steps to run the project locally on your machine.

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Shopper-Spectrum.git
cd Shopper-Spectrum
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Add the Dataset
Ensure that the raw dataset file `OnlineRetail.csv` is located in the root directory of the project.

### 4. Generate the Models
Before running the web app, you must process the data and generate the machine learning models. This step builds the similarity matrices, association rules, and customer profiles.
```bash
python generate_models.py
```
*(Note: Depending on your hardware, this may take a minute or two as it processes hundreds of thousands of transactions).*

### 5. Launch the Application
Once the `.pkl` files are generated in your directory, start the Streamlit server:
```bash
streamlit run Shopper_Spectrum_app.py
```

---

## 💡 Future Enhancements
*   Integration with an SQL database (like SQLite or DuckDB) to eliminate memory limitations for massive datasets.
*   Implementation of Customer Lifetime Value (CLV) predictions using BG/NBD models.
*   Time-series forecasting for inventory demand.