import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Set page title
st.set_page_config(page_title="Epileptic Seizure Detection", layout="centered")

# Load the trained model
model = load_model("seizure_detector1.keras")

# Title and description
st.title("Epileptic Seizure Detection ")
st.markdown("Upload an EEG CSV file, pick one sample, and predict seizure presence.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your EEG CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write(f"Total Samples in File: {len(df)}")
        # st.dataframe(df.head())  # Show preview

        # Filter only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Select only the first 178 columns expected by the model
        if numeric_df.shape[1] < 178:
            st.error("❌ Error: This file has fewer than 178 numeric features. Cannot proceed.")
        else:
            input_df = numeric_df.iloc[:, :178]  # select first 178 numeric columns

            # Sample selection
            index = st.number_input("🔢 Select row/sample number", min_value=0, max_value=len(input_df) - 1, step=1)

            if st.button("🔍 Predict"):
                # Extract the sample and reshape for prediction
                sample = input_df.iloc[index].values.astype(np.float32).reshape(1, -1)

                # Predict
                prediction = model.predict(sample)

                # Display result
                if prediction[0][0] > 0.5:
                    st.error("⚠️ Seizure Detected")
                else:
                    st.success("✅ No Seizure Detected")

    except Exception as e:
        st.error(f"❌ Error: {e}")
