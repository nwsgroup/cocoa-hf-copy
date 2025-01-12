from huggingface_hub import login
import matplotlib.pyplot as plt
from collections import Counter
import os
import sys
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HF_API_KEY")

if api_key:
    login(api_key)
else:
    raise ValueError("HF_API_KEY environment variable not found!")

from datasets import load_dataset

# Count the occurrences of each class in the training set
label_counts = Counter(dataset["train"]["label"])

# Convert label IDs to their string names (if applicable)
label_names = dataset["train"].features["label"].names
label_counts_named = {label_names[label]: count for label, count in label_counts.items()}

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.bar(label_counts_named.keys(), label_counts_named.values(), color='skyblue')
plt.xlabel("Class Labels")
plt.ylabel("Number of Samples")
plt.title("Class Distribution in SemilleroCV/Cocoa-dataset")
plt.xticks(rotation=45)
plt.show()
