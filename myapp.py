import streamlit as st
import pandas as pd
import joblib

header = st.beta_container()
dataset = st.beta_container()
model = st.beta_container()

with header:
    st.title("Poisonous Plant Identifer")
    st.write("This project is a machine learning model that predicts if an image of a plant is poisonous or not.")

with model:
    st.header("Run Model Option 1")
    if st.button("Run Model"):
    
        # unpickle the model
        model = joblib.load("Model/final_model.pkl")
        
        # Store inputs into dataframe
        X = pd.DataFrame([[height, weight, eyes]], 
                        columns = ["Height", "Weight", "Eyes"])
        X = X.replace(["Brown", "Blue"], [1, 0])
        
        # Get prediction
        prediction = model.predict(X)[0]
        
        # Output prediction
        st.text(f"This instance is a {prediction}")