from datasets import load_dataset

# Load the 1K-scale dataset
dataset = load_dataset("THUDM/ImageRewardDB", "1k",trust_remote_code=True)

for batch in dataset:
    print(batch)