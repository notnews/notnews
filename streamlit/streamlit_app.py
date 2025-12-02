import pandas as pd

import streamlit as st
from notnews import soft_news_url_cat_us

# Set app title
st.title("Not News")

# Add a file uploader widget for the user to upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Add a button to trigger the transformation
if st.button("Transform Data"):
    # Check if a file was uploaded
    if uploaded_file is not None:
        # Read the uploaded CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Use the package to transform the DataFrame
        transformed_df = soft_news_url_cat_us(df, col="url")

        # Display the transformed DataFrame as a table
        st.write(transformed_df)
    else:
        # Display an error message if no file was uploaded
        st.error("Please upload a CSV file to transform.")
