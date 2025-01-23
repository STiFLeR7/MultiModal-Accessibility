import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


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

        # Load annotations
        with open(annotations_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        print(f"Total annotations loaded: {len(annotations)}")

        # Debugging filtering logic
        valid_annotations = []
        invalid_annotations = []

        for ann in annotations:
            image_path = os.path.join(image_dir, ann.get("image_id", ""))
            if "image_id" in ann and "question" in ann and os.path.exists(image_path):
                valid_annotations.append(ann)
            else:
                invalid_annotations.append(ann)

        print(f"Valid annotations: {len(valid_annotations)}")
        print(f"Invalid annotations: {len(invalid_annotations)}")

        if invalid_annotations:
            print("\nSample invalid annotation:")
            print(invalid_annotations[0])

        # Assign valid annotations
        self.annotations = valid_annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get annotation details
        annotation = self.annotations[idx]

        # Debugging: Check annotation keys
        print(f"Annotation keys: {list(annotation.keys())}")

        # Safely access annotation fields
        image_path = os.path.join(self.image_dir, annotation.get("image_id", ""))
        question = annotation.get("question", "")
        answers = annotation.get("answers", [])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # Resize to match model input requirements

        # Tokenize the question
        tokenized_question = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Prepare output
        return {
            "image": image,
            "input_ids": tokenized_question["input_ids"].squeeze(0),
            "attention_mask": tokenized_question["attention_mask"].squeeze(0),
            "answers": answers,
        }


def get_vizwiz_dataloader(annotations_file, image_dir, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the VizWiz dataset.
    Args:
        annotations_file (str): Path to the annotations JSON file.
        image_dir (str): Path to the directory containing images.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = VizWizDataset(annotations_file, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
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
        print(f"- Image batch size: {len(batch['image'])}")
        print(f"- Tokenized question shape: {batch['input_ids'].shape}")
        print(f"- Sample answers: {batch['answers'][0]}")
        break  # Only test the first batch


if __name__ == "__main__":
    main()
