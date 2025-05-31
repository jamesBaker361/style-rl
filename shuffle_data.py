from datasets import load_dataset, DatasetDict
import random

for embedding in ["dino","ssl","siglip2","clip"]:
    for data in ["league_captioned_tile","league_captioned_splash","coco_captioned","art_coco_captioned","celeb_captioned"]:
        name=f"jlbaker361/{embedding}-{data}"
        # Load dataset from Hugging Face Hub
        dataset = load_dataset(name, split="train")  # You can change "imdb" and "train" to your dataset and split

        # Shuffle the dataset
        shuffled_dataset = dataset.shuffle(seed=42)
        shuffled_dataset.push_to_hub(name)
