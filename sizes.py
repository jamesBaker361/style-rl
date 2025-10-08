from datasets import load_dataset
import numpy as np

for embedding in ["clip","dino","ssl","siglip2"]:
    path=f"jlbaker361/{embedding}-league_captioned_splash-20"
    data=load_dataset(path,split="train")
    for row in data:
        break
    embedding_array=np.array(row["embedding"])
    print("\n\n",embedding,embedding_array.size)