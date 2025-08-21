import numpy as np
from sklearn.decomposition import PCA
from datasets import load_dataset
import time

start=time.time()
data=load_dataset("jlbaker361/clip-league_captioned_splash-1000",split="train")
for row in data:
    break


X_data=[np.array(row["embedding"]) for row in data]
print(X_data[0].shape)

pca_object=PCA(n_components=100)
pca_object.fit(X_data)
end=time.time()

print("elpased",end-start)

pca_object.transform(X_data[0])