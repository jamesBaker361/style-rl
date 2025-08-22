import numpy as np
from sklearn.decomposition import PCA
from datasets import load_dataset

for embedding in ["clip","dino","ssl","siglip2"]:
    data=load_dataset(f"jlbaker361/{embedding}-league_captioned_splash-1000",split="train")
    X_data=[np.array(row["embedding"])[0][0] for row in data]
    pca_object=PCA(n_components=100)
    pca_object.fit(X_data)
    np.savez(f"{embedding}_pca.npz",components_=pca_object.components_, explained_variance_=pca_object.explained_variance_)