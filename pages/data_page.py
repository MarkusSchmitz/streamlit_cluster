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
    page_title="Data",
    page_icon="👋",
)

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
        
    # check if the name already exists
    if name in [d["name"] for d in data]:
        st.error("Name already exists, please wait for the cluster analysis to update")
        
    else:

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