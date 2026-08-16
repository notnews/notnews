import pandas as pd

import streamlit as st
from notnews import classify_by_url

st.title("Not News")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if st.button("Transform Data"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        transformed_df = classify_by_url(df, url_col="url", region="us")
        st.write(transformed_df)
    else:
        st.error("Please upload a CSV file to transform.")
