import streamlit as st
from dotenv import load_dotenv
import os

st.set_page_config(
    page_title="Entry",
    page_icon="👋",
)



st.markdown("""
This is a project to cluster students into groups based on their data.

## Goals
This serves two purposes:
1. A fun way to get to know each other and to create groups for the project.
2. A simple yet powerful example of how to use AI to solve a real-world problem.

## How it works
1. Each student enters their data.
2. The data is used to create a semantic embedding.
3. The semantic embeddings are clustered using K-Means.
4. An additional algorithm is used to evenly distribute the students among the clusters.
5. The embeddings are reduced to two dimensions using PCA.
6. The clusters are displayed using a scatter plot.

## How to use it
Enter your data by clicking on the data page.
Alternatively, you can scan the QR code below to go directly to the data page.


## Attention:
Please only enter your data once.
If you enter your data multiple times, the clusters will be inaccurate.

The data you enter will be able to be seen by the other students.
Please be mindful of what you enter.
""")

#st.image("qr.png")

if st.text_input("Clear Data", type="password") == "ITM2023/24":
    with open("data.json", "w") as f:
        f.write("[]")
    st.success("Data cleared")
