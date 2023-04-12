# from tsne_torch import TorchTSNE as TSNE
import json
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

def get_tsne(dataset, verbose=False):
    if hasattr(dataset, 'indices'):
        df = dataset.dataset.df.iloc[dataset.indices]
    else:
        df = dataset.df
    column = 'probs'
    try:
        sample_vector = json.loads(df[column].values[0])
    except TypeError:
        column = 'embedding'
        sample_vector = json.loads(df[column].values[0])
    
    emb_size = len(sample_vector)
    X = torch.zeros((df.shape[0], emb_size))
    field_index = tuple(df.columns).index(column) + 1 # plus one for "index column"

    for r, row in enumerate(df.itertuples()):
        X[r] = torch.tensor(json.loads(row[field_index]))

    tsne = TSNE(n_components=2, verbose=verbose, perplexity=40, n_iter=300, learning_rate='auto', init='pca', random_state=42)

    X_emb = tsne.fit_transform(X)

    scaler = MinMaxScaler()
    X_emb = scaler.fit_transform(X_emb)

    return X_emb

def plot_tsne(X_emb, labels):
    import matplotlib.pyplot as plt
    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=labels, cmap=plt.cm.tab20c)
    plt.savefig("/tmp/tsne.png")
    plt.close()