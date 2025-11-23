from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset("imagefolder", data_dir="data")
print("Dataset loaded.")
print("Train split columns:", dataset["train"].column_names)
print("Test split columns:", dataset["test"].column_names)

if "file" in dataset["train"].column_names:
    print("Renaming 'file' to 'image'...")
    dataset = dataset.rename_column("file", "image")
    print("Renamed.")
    print("Train split columns after rename:", dataset["train"].column_names)

print("Checking first example keys...")
try:
    print(dataset["train"][0].keys())
except Exception as e:
    print("Error accessing first example:", e)
