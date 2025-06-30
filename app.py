import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(page_title="Facial Verification System", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    with open("best_face_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Page title
st.title("🔐 Facial Identity Verification App")
st.markdown("Upload a single-row CSV with facial features to predict identity.")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Drop label column if present
        if 'Label' in input_df.columns:
            input_df = input_df.drop(columns=['Label'])

        st.subheader("🔎 Uploaded Data")
        st.dataframe(input_df)

        # Predict
        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)

        st.success(f"✅ Predicted Label: {prediction[0]}")

        # Show probabilities
        prob_df = pd.DataFrame(probabilities, columns=model.classes_)
        prob_df = prob_df.T.rename(columns={0: "Probability"})
        st.subheader("🔢 Prediction Probabilities")
        st.dataframe(prob_df)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("Please upload a CSV file to proceed.")

# Footer
st.markdown("---")
