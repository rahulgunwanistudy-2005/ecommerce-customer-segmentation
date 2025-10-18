import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="E-commerce Customer Dashboard",
    page_icon="🛒",
    layout="wide"
)

# --- Load Data and Artifacts ---
@st.cache_data # Caches the data to avoid reloading on every interaction
def load_data():
    # Load the final RFM dataframe with clusters and churn info
    # NOTE: You'll need to save this from your notebook first!
    # Example: rfm_df.to_csv('rfm_customer_data.csv')
    df = pd.read_csv('rfm_customer_data.csv')
    return df

@st.cache_resource # Caches the model to avoid reloading
def load_artifacts():
    # Load the saved model and scaler
    artifacts = joblib.load('churn_artifacts.joblib')
    return artifacts

df = load_data()
artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']

# --- Helper Function ---
def plot_segment_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Cluster', data=df, ax=ax, palette='viridis')
    ax.set_title('Customer Segment Distribution')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Number of Customers')
    # Add labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    return fig

# --- Dashboard UI ---
st.title("🛒 Dynamic Customer Segmentation & Churn Prediction")

# --- Overview Section ---
st.header("Overall Customer Insights")
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Customers", df.shape[0])
    st.pyplot(plot_segment_distribution())

with col2:
    st.subheader("Segment Profiles (Averages)")
    cluster_profile = df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Churn': 'mean' # Average churn rate per segment
    }).round(2)
    st.dataframe(cluster_profile)


# --- Churn Prediction Section ---
st.header("🔮 Predict Customer Churn")

st.markdown("Enter customer data to predict their churn probability.")

# Create input fields for user
recency = st.number_input("Recency (Days since last purchase)", min_value=0, step=1)
frequency = st.number_input("Frequency (Total number of purchases)", min_value=1, step=1)
monetary = st.number_input("Monetary (Total spend)", min_value=0.0, step=10.0)

# The predict button
if st.button("Predict Churn"):
    # Find the cluster for the new customer data
    # We create a temporary DataFrame to scale the input
    temp_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
    temp_scaled = scaler.transform(temp_df)
    
    # We need the original K-Means model to predict the cluster
    # NOTE: You need to save the kmeans model from your notebook as well!
    # Example: joblib.dump(kmeans, 'kmeans_model.joblib')
    kmeans = joblib.load('kmeans_model.joblib') # You'll need to save this
    cluster_pred = kmeans.predict(temp_scaled)[0]

    # Now make the final churn prediction
    input_data = [[recency, frequency, monetary, cluster_pred]]
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1] # Probability of churn

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"This customer is likely to CHURN (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"This customer is likely to STAY (Probability of Churn: {prediction_proba:.2f})")

    st.info(f"The customer is predicted to belong to **Cluster {cluster_pred}**.")