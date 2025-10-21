# Dynamic Customer Segmentation & Churn Prediction Dashboard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end data science project that segments e-commerce customers using unsupervised machine learning and predicts churn with a supervised classification model. The results are presented in an interactive Streamlit dashboard designed for a marketing team.

Live Demo: https://customer-churn-prediction--dashboard.streamlit.app

---

## Dashboard Preview



---

## ## Project Overview

This project moves beyond a simple, one-size-fits-all churn prediction. It first identifies distinct customer personas based on their purchasing behavior using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering. These segments could be, for example, "High-Value Champions," "At-Risk Customers," or "Newbies."

Following segmentation, a logistic regression model is trained to predict the likelihood of churn for any given customer. The entire pipeline—from data cleaning to prediction—is wrapped in an interactive and user-friendly web dashboard, allowing non-technical stakeholders to gain insights and make data-driven decisions for targeted marketing campaigns.

## ## Key Features

- **Data Cleaning & Preprocessing:** Handles missing values, duplicates, and cancellations from a raw transactional dataset.
- **Feature Engineering:** Creates powerful **RFM (Recency, Frequency, Monetary)** metrics for each customer.
- **Unsupervised Learning:** Employs **K-Means clustering** to segment customers into distinct behavioral groups. The optimal number of clusters is determined using the Elbow Method.
- **Customer Personas:** Each customer segment is profiled and analyzed to create actionable business insights.
- **Supervised Learning:** A **Logistic Regression** model is trained to predict customer churn based on their RFM values and segment.
- **Interactive Dashboard:** Built with **Streamlit**, the dashboard allows users to view segment distributions, analyze profiles, and get on-demand churn predictions for hypothetical customers.

---

## ## Tech Stack

- **Programming Language:** Python 3.10+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Web App/Dashboard:** Streamlit
- **Version Control:** Git & GitHub

---

## ## Methodology

The project follows a standard data science workflow:

1.  **Data Cleaning & EDA:** The raw [Online Retail II dataset from UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii) was loaded. Extensive cleaning was performed, including handling over 100,000 missing `Customer ID`s and removing transactional returns.
2.  **Feature Engineering:** The core of the analysis, where the cleaned data was transformed to create the RFM metrics for each unique customer.
3.  **Customer Segmentation (Unsupervised):** The scaled RFM features were fed into a K-Means algorithm. The Elbow Method was used to identify an optimal `k=4` clusters. These clusters were then analyzed to create meaningful personas (e.g., Champions, At-Risk, etc.).
4.  **Churn Prediction (Supervised):** A "Churn" label was engineered based on a recency threshold (>90 days). The data was split into training and testing sets, and a logistic regression model was trained to predict this label.
5.  **Dashboard Development:** The trained K-Means model, scaler, and logistic regression model were saved as `.joblib` artifacts. A Streamlit script (`app.py`) was created to load these artifacts and build an interactive user interface for exploration and prediction.

---

## ## How to Run Locally

To run this project on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/rahulgunwanistudy-2005/ecommerce-customer-segmentation.git](https://github.com/rahulgunwanistudy-2005/ecommerce-customer-segmentation.git)
    cd ecommerce-customer-segmentation
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

---

## ## Project Structure
```
ecommerce-customer-segmentation/
├── app.py                  # The main Streamlit dashboard script
├── churn_artifacts.joblib  # Saved churn model and scaler
├── kmeans_model.joblib     # Saved K-Means clustering model
├── rfm_customer_data.csv   # The final processed data for the app
├── notebooks/
│   └── 01_data_exploration.ipynb # Jupyter notebook with all analysis
├── data/
│   └── online_retail_II.xlsx   # (Not committed) Raw dataset
├── requirements.txt        # Python dependencies
└── README.md               # You are here!
```