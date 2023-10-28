# Streamlit app that clusters the data and displays the results
# It also has an input for people to enter their own data and see which cluster they belong to

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
import json
from aleph_alpha_client import Client, Prompt, SemanticEmbeddingRequest, SemanticEmbeddingResponse, SemanticRepresentation
import os

# Title
st.title("Cluster Analysis")

# simple db to store the data which is saved in data.json
# data is a list of dictionaries
# each dictionary is a row in the data

# two tabs: 1. Enter Data 2. Cluster Analysis
enter_data, cluster_analysis = st.columns(2)

# Enter Data
with enter_data:
    # form to enter data
    st.header("Enter Data")
    st.write("Enter the data you want to cluster")

    name = st.text_input("Your Name", key="name")
    enjoy = st.text_input("What do you enjoy most during your studies?", key="enjoy")
    least = st.text_input("What do you enjoy least during your studies?", key="least")
    free_time = st.text_input("What do you love to do in your free time?", key="free_time")
    food = st.text_input("What is your favorite food?", key="food")
    truth = st.text_input("What is a truth that too few people agree with you on?", key="truth")

    # save button
    if st.button("Save"):
        # append the data to the data.json file
        # load the data
        with open("data.json", "r") as f:
            data = json.load(f)

        entry = f"""
            {name} enjoys {enjoy} the most during their studies,
            but they enjoy {least} the least.
            In their free time, they like to {free_time}.
            Their favorite food is {food}.
            A truth that too few people agree with them on is {truth}.
            """

        request = SemanticEmbeddingRequest(
            prompt=Prompt.from_text(entry),
            representation=SemanticRepresentation.Symmetric,
            compress_to_size=128)

        # append the data to the data list
        data.append({
            "name": name,
            "entry": entry,
            "embedding": Client(token=os.getenv("AA_TOKEN")).semantic_embed(request=request, model="luminous-base").embedding
        })

        # save the data
        with open("data.json", "w") as f:
            json.dump(data, f)

        # success message
        st.success("Data saved successfully")

# Cluster Analysis
with cluster_analysis:

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
    




    

