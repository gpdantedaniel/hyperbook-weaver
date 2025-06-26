from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize
from collections import defaultdict
import umap.umap_ as umap
import plotly.express as px
import pandas as pd

model_name = 'Qwen/Qwen3-Embedding-0.6B'

def get_keywords(df):
    """
    Extracts keywords for each paper
    and appends them to each row.
    """

    keybert = KeyBERT(model=model_name)
    grouped_keywords = []

    for content in df['content']:
        pairs = keybert.extract_keywords(
            content,
            top_n=20,
            stop_words='english',
            nr_candidates=40
        )
        keywords = [pair[0] for pair in pairs]
        grouped_keywords.append(keywords)
    
    return grouped_keywords


embedding_model = SentenceTransformer(model_name)
clusterer = HDBSCAN(min_cluster_size=2, metric='euclidean', algorithm='best')

def cluster_keywords(keywords: list[str]) -> dict:
    """
    Clusters relevant keywords into semantically
    related clusters and returns a map
    """
    embeddings = embedding_model.encode(keywords, show_progress_bar=True)
    embeddings = normalize(embeddings, norm='l2')
    clusters = clusterer.fit_predict(embeddings)
    cluster_map = defaultdict(list)

    for keyword, cluster_id in zip(keywords, clusters):
        if cluster_id != -1: # Skip noise if using HDBSCAN
            cluster_map[int(cluster_id)].append(keyword)

    print(f'Identified {len(cluster_map)} clusters')
    return cluster_map


def name_clusters(cluster_map: dict) -> dict:
    """
    Maps each cluster identifier to a semantically relevant name
    """
    cluster_names = {}

    for cluster_id in cluster_map.keys():
        name = f'cluster_{cluster_id}'
        cluster_names[cluster_id] = name

    return cluster_names

def display_clusters(cluster_map: dict, cluster_names: dict):
    """
    Displays clusters and their member keywords
    """

    flat_keywords, flat_ids = flatten_cluster_map(cluster_map)
    flat_names = [cluster_names[cluster_id] for cluster_id in flat_ids]

    embeddings = embedding_model.encode(flat_keywords)
    embeddings_2d = reducer.fit_transform(embeddings)

    coords = pd.DataFrame({
        'keyword': flat_keywords,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'cluster': flat_names
    })
        
    return px.scatter(
        coords, 
        x='x', 
        y='y',
        color=flat_names,
        hover_data=['keyword'],
        title='Topic clusters'
    )

def flatten_cluster_map(cluster_map: dict) -> tuple:
    """
    Flattens a cluster map and returns two equally long lists
    """
    flat_keywords = [
        kw 
        for group in cluster_map.values() 
        for kw in group
    ]
    flat_ids = [
        mapped_id 
        for cluster_id in cluster_map.keys() 
        for mapped_id in [cluster_id] * len(cluster_map[cluster_id]) 
    ]
    return flat_keywords, flat_ids

reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine', random_state=42)


if __name__ == '__main__':
    # Run the script
    None
