from datasets import load_dataset

# Load the 1K-scale dataset
dataset = load_dataset("THUDM/ImageRewardDB", "1k")

for batch in dataset:
    break

print(batch)