import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Page config
st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")

# Load model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Single Prediction", "Batch Prediction", "Analysis"])

# Home Page
if page == "Home":
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown("""
    Welcome to the Credit Card Fraud Detection Application. This tool uses a **Random Forest** machine learning model 
    trained on anonymized credit card transaction data to identify potentially fraudulent activities.
    
    ### How it works:
    1. **Single Prediction**: Enter transaction details manually to check for fraud.
    2. **Batch Prediction**: Upload a CSV file with multiple transactions.
    3. **Analysis**: View insights and performance metrics of the model.
    
    *Data Source: Kaggle Credit Card Fraud Detection Dataset.*
    """)
    st.image("https://images.unsplash.com/photo-1563013544-824ae1b704d3?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", caption="Secure Transactions")

# Single Prediction
elif page == "Single Prediction":
    st.title("🔍 Single Transaction Check")
    st.write("Enter the details below to predict if a transaction is fraudulent.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time = st.number_input("Time (Seconds from first transaction)", value=0.0)
        amount = st.number_input("Transaction Amount ($)", value=0.0)
    
    v_features = {}
    for i in range(1, 29):
        with [col1, col2, col3][i % 3]:
            v_features[f'V{i}'] = st.number_input(f'V{i} (Anonymized Feature)', value=0.0)
            
    if st.button("Predict"):
        # Prepare input
        input_data = pd.DataFrame([[time] + [v_features[f'V{i}'] for i in range(1, 29)] + [amount]], 
                                  columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])
        
        # Scale
        input_data['Amount'] = scaler.transform(input_data['Amount'].values.reshape(-1, 1))
        # (Assuming Time was also scaled in training script)
        input_data['Time'] = scaler.transform(input_data['Time'].values.reshape(-1, 1))
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        if prediction == 1:
            st.error(f"🚨 **Warning!** This transaction is likely **FRAUDULENT** (Probability: {probability:.2%})")
        else:
            st.success(f"✅ This transaction appears to be **LEGITIMATE** (Fraud Probability: {probability:.2%})")

# Batch Prediction
elif page == "Batch Prediction":
    st.title("📂 Batch Transaction Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file containing transaction data", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(data.head())
        
        if st.button("Run Batch Prediction"):
            # Check columns
            required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            if all(col in data.columns for col in required_cols):
                process_data = data[required_cols].copy()
                process_data['Amount'] = scaler.transform(process_data['Amount'].values.reshape(-1, 1))
                process_data['Time'] = scaler.transform(process_data['Time'].values.reshape(-1, 1))
                
                preds = model.predict(process_data)
                probs = model.predict_proba(process_data)[:, 1]
                
                data['Is_Fraud'] = preds
                data['Fraud_Probability'] = probs
                
                st.write("### Prediction Results")
                st.dataframe(data)
                
                fraud_count = sum(preds)
                st.metric("Fraudulent Transactions Detected", fraud_count)
                
                # Download results
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")
            else:
                st.error("Uploaded file is missing required columns (Time, V1-V28, Amount).")

# Analysis Page
elif page == "Analysis":
    st.title("📊 Model Insights")
    
    st.write("""
    The model was trained using a **Random Forest Classifier** with **SMOTE** (Synthetic Minority Over-sampling Technique) 
    to handle the extreme class imbalance in the dataset.
    """)
    
    # Feature Importance
    importances = model.feature_importances_
    features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    feat_importances = pd.Series(importances, index=features).sort_values(ascending=False).head(10)
    
    st.write("### Top 10 Most Important Features")
    fig, ax = plt.subplots()
    sns.barplot(x=feat_importances.values, y=feat_importances.index, ax=ax, palette="viridis")
    st.pyplot(fig)
    
    st.info("The features V1-V28 are result of PCA transformation for privacy reasons.")
