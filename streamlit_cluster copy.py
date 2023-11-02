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
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# load the environment variables
load_dotenv()


def reorder_clusters(df):
    """
    Reorders the clusters so that they have the same number of elements
    
    Args:
        df (pd.DataFrame): The dataframe containing the data

    Returns:
        pd.DataFrame: The reordered dataframe
    """




    

st.title("Cluster Analysis")

# simple db to store the data which is saved in data.json
# data is a list of dictionaries
# each dictionary is a row in the data

# simple admin or user login
# if admin, show the data
# if user, show the form to enter data


    # two tabs: 1. Enter Data 2. Cluster Analysis
enter_data, cluster_analysis = st.tabs(["Enter Data", "Cluster Analysis"])

    # Enter Data
with enter_data:

    st.session_state["submitted"] = False
    # form to enter data
    st.header("Enter Data")
    st.write("Enter the data you want to cluster")
    with st.container():
        name = st.text_input("Your Name")
        enjoy = st.text_input("What do you enjoy most during your studies?")
        least = st.text_input("What do you enjoy least during your studies?")
        free_time = st.text_input("What do you love to do in your free time?")
        food = st.text_input("What is your favorite food?")
        truth = st.text_input("What is a truth that too few people agree with you on?")

        # submit button
        submit_entry = st.button("Submit")

        if submit_entry:
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
            
            if name not in [d["name"] for d in data]:


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
                st.session_state["submitted"] = True

                st.success("Data saved successfully")    
            else:
                st.error("Name already exists")

# Cluster Analysis
with cluster_analysis:



    plot = st.empty()
    plot_button = st.button("Plot")

    if plot_button:

        # load the data
        with open("data.json", "r") as f:
            data = json.load(f)

        # convert the data to a dataframe
        df = pd.DataFrame(data)

        # remove the entry column
        df = df.drop(columns=["entry"])


        # get the embeddings
        embeddings = np.array([d["embedding"] for d in data])

        # get the number of clusters
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5, step=1)

        # cluster the data
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embeddings)

        # add the cluster labels to the dataframe
        df["cluster"] = kmeans.labels_

        # check if all clusters have the same number of elements
        cluster_lens = [len(df[df["cluster"] == i]) for i in range(n_clusters)]
        if len(set(cluster_lens)) != 1:

            print("reordering clusters")

            cluster_centers = kmeans.cluster_centers_

            # manually assign the entries to the clusters
            n_elements_per_cluster = len(df) // n_clusters
            elements_to_assign = df.to_dict("records")
            assigned_elements = []

            for i in range(n_clusters):
                # get thne n elements closest to the cluster center
                cluster_center = cluster_centers[i]
                for j in range(n_elements_per_cluster):
                    # get the closest element by cosine distance
                    distances = [cosine(cluster_center, e["embedding"]) for e in elements_to_assign]
                    closest_element = np.argmin(distances)
                    element = elements_to_assign[closest_element]
                    element["cluster"] = i
                    # add the element to the assigned elements
                    assigned_elements.append(element)
                    # remove the element from the elements to assign
                    elements_to_assign.pop(closest_element)
            
            # if there are still elements to assign, assign them randomly
            if len(elements_to_assign) > 0:
                for e in elements_to_assign:
                    e["cluster"] = np.random.randint(n_clusters)
                    assigned_elements.append(e)

            # convert the assigned elements to a dataframe
            df = pd.DataFrame(assigned_elements)

        

            

        # use pca to reduce the dimensionality of the embeddings
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        positions = pca.fit_transform(embeddings).tolist()
        df["x"] = [p[0] for p in positions]
        df["y"] = [p[1] for p in positions]

        # display the dataframe
        st.dataframe(df)

        # display the clusters
        fig = px.scatter(df, x="x", y="y", color="cluster", hover_data=["name", "cluster"])

        # plot the centroids
        centroids_reduced = pca.transform(kmeans.cluster_centers_)
        centroids = pd.DataFrame(centroids_reduced, columns=["x", "y"])        

        fig.add_scatter(x=centroids["x"], y=centroids["y"], mode="markers", marker=dict(color="black", size=20))

        st.plotly_chart(fig)

        # do the same with tsne
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2)
        positions = tsne.fit_transform(embeddings).tolist()
        df["x"] = [p[0] for p in positions]
        df["y"] = [p[1] for p in positions]

        # display the clusters
        fig = px.scatter(df, x="x", y="y", color="cluster", hover_data=["name", "cluster"])

        # plot the centroids
        centroids_reduced = tsne.transform(kmeans.cluster_centers_)
        centroids = pd.DataFrame(centroids_reduced, columns=["x", "y"])

        fig.add_scatter(x=centroids["x"], y=centroids["y"], mode="markers", marker=dict(color="black", size=20))

        st.plotly_chart(fig)

