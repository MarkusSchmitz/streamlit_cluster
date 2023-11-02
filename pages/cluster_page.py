import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
import json
from aleph_alpha_client import Client, Prompt, SemanticEmbeddingRequest, SemanticEmbeddingResponse, SemanticRepresentation
import os
from dotenv import load_dotenv

# load the environment variables
load_dotenv()

st.set_page_config(
    page_title="Cluster",
    page_icon="👋",
)
if st.text_input("password", type="password") == "ITM2023/24":
    with open("data.json", "r") as f:
            data = json.load(f)

    if len(data) > 6:
        # Cluster Analysis

        # load the data
        with open("data.json", "r") as f:
            data = json.load(f)

        # convert the data to a dataframe
        df = pd.DataFrame(data)

        # put the names into the dataframe
        df["name"] = df["name"].apply(lambda x: x.capitalize())

        # get the embeddings
        embeddings = np.array([d["embedding"] for d in data])

        # get the number of clusters
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3, step=1)

        # cluster the data
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embeddings)

        # add the cluster labels to the dataframe
        df["cluster"] = kmeans.labels_

        # use pca to reduce the dimensionality of the embeddings
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        df["embedding"] = pca.fit_transform(embeddings).tolist()

        # display the dataframe
        st.dataframe(df)

        # display the clusters
        fig = px.scatter(df, x="embedding", color="cluster", hover_data=["name", "entry"])
        st.plotly_chart(fig)

    else:
        st.info("Not enough data to cluster yet, please wait for more data to be added")
        
else:
    st.markdown("""Enter the password to view the cluster analysis""")