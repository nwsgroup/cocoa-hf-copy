import json
from huggingface_hub import login
import os
import datasets
from transformers import AutoFeatureExtractor, AutoModel
import torch
from annoy import AnnoyIndex
from renumics import spotlight
from cleanlab.outlier import OutOfDistribution
import numpy as np
import pandas as pd
import sys
import requests
import umap

def extract_embeddings(model, feature_extractor, image_name="image"):
    """
    Utility to compute embeddings.
    Args:
        model: huggingface model
        feature_extractor: huggingface feature extractor
        image_name: name of the image column in the dataset
    Returns:
        function to compute embeddings
    """
    device = model.device
    def pp(batch):
        images = batch[image_name]
        inputs = feature_extractor(
            images=[x.convert("RGB") for x in images], return_tensors="pt"
        ).to(device)
        embeddings = model(**inputs).last_hidden_state[:, 0].cpu()
        return {"embedding": embeddings}
    return pp

def huggingface_embedding(
    df,
    image_name="image",
    modelname="CristianR8/mobilenet_large-model",
    batched=True,
    batch_size=24,
):
    """
    Compute embeddings using huggingface models.
    Args:
        df: dataframe with images
        image_name: name of the image column in the dataset
        modelname: huggingface model name
        batched: whether to compute embeddings in batches
        batch_size: batch size
    Returns:
        new dataframe with embeddings
    """
    # initialize huggingface model
    feature_extractor = AutoFeatureExtractor.from_pretrained(modelname)
    if "vit" in modelname:
        model = AutoModel.from_pretrained(modelname, output_hidden_states=True, add_pooling_layer=False)
    else:
        model = AutoModel.from_pretrained(modelname, output_hidden_states=True)
    # create huggingface dataset from df
    dataset = datasets.Dataset.from_pandas(df).cast_column(image_name, datasets.Image())
    # compute embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_fn = extract_embeddings(model.to(device), feature_extractor, image_name)
    updated_dataset = dataset.map(extract_fn, batched=batched, batch_size=batch_size)
    df_temp = updated_dataset.to_pandas()
    df_emb = pd.DataFrame()
    df_emb["embedding"] = df_temp["embedding"]
    return df_emb


DATASET_NAME = "SemilleroCV/Cocoa-dataset"
dataset = datasets.load_dataset(DATASET_NAME, split='all')

df = dataset.to_pandas()

ft_model_name = "Factral/convnext_xlarge-cocoa"

df_emb = huggingface_embedding(df, modelname=ft_model_name)
df = pd.concat([df, df_emb], axis=1)


embeddings = np.stack(df["embedding"].to_numpy())

reducer = umap.UMAP()
reduced_embedding = reducer.fit_transform(embeddings)
df["embedding_reduced"] = np.array(reduced_embedding).tolist()

print(np.array(reduced_embedding.shape))

df_show = df.drop(columns=["embedding"])

layout_url = "https://raw.githubusercontent.com/Renumics/spotlight/main/playbook/rookie/embedding_layout.json"
response = requests.get(layout_url)
layout = spotlight.layout.nodes.Layout(**json.loads(response.text))
spotlight.show(
    df_show,
    dtype={"image": spotlight.Image, "embedding_reduced": spotlight.Embedding},
    layout=layout,
    port=7007
)