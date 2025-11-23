
from transformers import AutoImageProcessor, ResNetForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
from torchvision import transforms
import torch
import evaluate
from transformers import get_scheduler
from torch.optim import AdamW
from datasets import Image


print("Finished Imports")

metric = evaluate.load("accuracy")
model_name = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(model_name, num_labels=3, ignore_mismatched_sizes=True)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expected input size
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # shift up to ±10%
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

def transform_train(examples):
    images = [train_transform(img.convert("RGB")) for img in examples["image"]]
    inputs = {"pixel_values": images, "labels": examples["label"]}
    return inputs

def transform_test(examples):
    images = [test_transform(img.convert("RGB")) for img in examples["image"]]
    inputs = {"pixel_values": images, "labels": examples["label"]}
    return inputs

dataset = load_dataset("imagefolder", data_dir="data")

device = torch.device("cuda")
model.to(device)


normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)

if "file" in dataset["train"].column_names:
    dataset = dataset.rename_column("file", "image")

dataset = dataset.cast_column("image", Image())

train_dataset = dataset["train"].with_transform(transform_train)
test_dataset = dataset["test"].with_transform(transform_test)

training_args = TrainingArguments(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=15,
    eval_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    logging_strategy="epoch",
    learning_rate=3e-4,
    report_to="none",
    remove_unused_columns=True,
    fp16=torch.cuda.is_available()
)

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1) 
    return metric.compute(predictions=predictions, references=labels)

scheduler = get_scheduler(
    "cosine",             
    optimizer = optimizer,
    num_warmup_steps = 500,     
    num_training_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)

from transformers import AutoImageProcessor, ResNetForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
from torchvision import transforms
import torch
import evaluate
from transformers import get_scheduler
from torch.optim import AdamW
from datasets import Image


print("Finished Imports")

metric = evaluate.load("accuracy")
model_name = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(model_name, num_labels=3, ignore_mismatched_sizes=True)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expected input size
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # shift up to ±10%
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

def transform_train(examples):
    images = [train_transform(img.convert("RGB")) for img in examples["image"]]
    inputs = {"pixel_values": images, "labels": examples["label"]}
    return inputs

def transform_test(examples):
    images = [test_transform(img.convert("RGB")) for img in examples["image"]]
    inputs = {"pixel_values": images, "labels": examples["label"]}
    return inputs

dataset = load_dataset("imagefolder", data_dir="data")

device = torch.device("cuda")
model.to(device)


normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)

if "file" in dataset["train"].column_names:
    dataset = dataset.rename_column("file", "image")

dataset = dataset.cast_column("image", Image())

train_dataset = dataset["train"].with_transform(transform_train)
test_dataset = dataset["test"].with_transform(transform_test)

training_args = TrainingArguments(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=15,
    eval_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    logging_strategy="epoch",
    learning_rate=3e-4,
    report_to="none",
    remove_unused_columns=True,
    fp16=torch.cuda.is_available()
)

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1) 
    return metric.compute(predictions=predictions, references=labels)

scheduler = get_scheduler(
    "cosine",             
    optimizer = optimizer,
    num_warmup_steps = 500,     
    num_training_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)

trainer.train()
print("Training completed.")


metrics = trainer.evaluate()
print("Model metrics:", metrics)

import json
import os
os.makedirs("trainer_output", exist_ok=True)
with open("trainer_output/full_history.json", "w") as f:
    json.dump(trainer.state.log_history, f)