import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="E-commerce Customer Dashboard",
    page_icon="🛒",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv('rfm_customer_data.csv')
    return df

@st.cache_resource
def load_artifacts():
    artifacts = joblib.load('churn_artifacts.joblib')
    return artifacts

df = load_data()
artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']

def plot_segment_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Cluster', data=df, ax=ax, palette='viridis')
    ax.set_title('Customer Segment Distribution')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Number of Customers')
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    return fig

st.title("🛒 Dynamic Customer Segmentation & Churn Prediction")

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
        'Churn': 'mean'
    }).round(2)
    st.dataframe(cluster_profile)


st.header("🔮 Predict Customer Churn")

st.markdown("Enter customer data to predict their churn probability.")

recency = st.number_input("Recency (Days since last purchase)", min_value=0, step=1)
frequency = st.number_input("Frequency (Total number of purchases)", min_value=1, step=1)
monetary = st.number_input("Monetary (Total spend)", min_value=0.0, step=10.0)

if st.button("Predict Churn"):
    kmeans = joblib.load('kmeans_model.joblib')
    
    temp_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
    temp_scaled = scaler.transform(temp_df)
    cluster_pred = kmeans.predict(temp_scaled)[0]

    input_data = [[recency, frequency, monetary, cluster_pred]]
    
    prediction_proba = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")

    st.progress(prediction_proba)

    prob_percentage = f"{prediction_proba*100:.2f}%"

    if prediction_proba < 0.3:
        st.success(f"Low Churn Risk: {prob_percentage}")
        st.write("This customer is likely to stay. A great candidate for loyalty programs!")
    elif prediction_proba < 0.7:
        st.warning(f"Medium Churn Risk: {prob_percentage}")
        st.write("This customer is on the fence. Consider sending a targeted promotion or follow-up.")
    else:
        st.error(f"High Churn Risk: {prob_percentage}")
        st.write("This customer is very likely to churn. Immediate action is recommended to retain them.")

    st.info(f"The customer is predicted to belong to **Cluster {cluster_pred}**.")