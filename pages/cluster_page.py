import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
import json
from aleph_alpha_client import Client, Prompt, SemanticEmbeddingRequest, SemanticEmbeddingResponse, SemanticRepresentation
import os
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

# load the environment variables
load_dotenv()

st.set_page_config(
    page_title="Cluster",
    page_icon="👋",
)
if st.text_input("password", type="password") == os.getenv("PASSWORD"):
    with open("data.json", "r") as f:
            data = json.load(f)

    if len(data) > 6:
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
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4, step=1)

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
            """
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
            """

    else:
        st.info("Not enough data to cluster yet, please wait for more data to be added")
        
else:
    st.markdown("""Enter the password to view the cluster analysis""")