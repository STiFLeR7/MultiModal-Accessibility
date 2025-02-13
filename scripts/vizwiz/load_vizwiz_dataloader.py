import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torchvision.transforms as T
import torch

class VizWizDataset(Dataset):
    def __init__(self, annotations_file, image_dir, tokenizer_model="bert-base-uncased", max_length=128):
        """
        Initialize the VizWiz dataset.
        Args:
            annotations_file (str): Path to the annotations JSON file.
            image_dir (str): Path to the directory containing images.
            tokenizer_model (str): Pretrained tokenizer model name.
            max_length (int): Maximum token length for text inputs.
        """
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_length = max_length

        # Define image transformation: Resize and convert to Tensor
        self.image_transform = T.Compose([
            T.Resize((224, 224)),  # Resize image to 224x224
            T.ToTensor(),         # Convert image to tensor
        ])

        # Load annotations
        with open(annotations_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        print(f"Total annotations loaded: {len(annotations)}")

        # Filter valid annotations
        valid_annotations = []
        for ann in annotations:
            image_path = os.path.join(image_dir, ann.get("image_id", ""))
            if "image_id" in ann and "question" in ann and os.path.exists(image_path):
                # Ensure default for answer if missing
                ann["answer"] = ann.get("answer", "unanswerable") or "unanswerable"
                valid_annotations.append(ann)

        print(f"Valid annotations: {len(valid_annotations)}")
        self.annotations = valid_annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # Safely access fields
        image_path = os.path.join(self.image_dir, annotation["image_id"])
        question = annotation["question"]
        answer = annotation["answer"]

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # Tokenize the question
        tokenized_question = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": tokenized_question["input_ids"].squeeze(0),
            "attention_mask": tokenized_question["attention_mask"].squeeze(0),
            "answer": answer,
        }


def collate_fn(batch):
    sanitized_batch = []
    for item in batch:
        if item["image"] is not None and item["input_ids"] is not None:
            sanitized_batch.append(item)
    if not sanitized_batch:
        raise ValueError("All items in the batch are invalid!")
    return {
        "image": torch.stack([item["image"] for item in sanitized_batch]),
        "input_ids": torch.stack([item["input_ids"] for item in sanitized_batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in sanitized_batch]),
        "answers": [item["answer"] for item in sanitized_batch],
    }


def get_vizwiz_dataloader(annotations_file, image_dir, batch_size=32, shuffle=True):

    dataset = VizWizDataset(annotations_file, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def main():
    # File paths
    annotations_file = "D:/MultiModal-Accessibility/preprocessed/VizWiz-2023/annotations/train.json"
    image_dir = "D:/MultiModal-Accessibility/preprocessed/VizWiz-2023/images/train"

    # Debugging paths
    print(f"Annotations file: {annotations_file}")
    print(f"Image directory: {image_dir}")

    # Create DataLoader
    dataloader = get_vizwiz_dataloader(annotations_file, image_dir)

    # Iterate through batches
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i + 1}:")
        for idx in range(min(5, len(batch["answers"]))):  # Display first 5 answers in the batch
            print(f"- Answer {idx + 1}: {batch['answers'][idx]}")
        if i >= 4:
            break

if __name__ == "__main__":
    main()
