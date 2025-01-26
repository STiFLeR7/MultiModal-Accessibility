import os
import json
import random
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torchvision.transforms as T
import matplotlib.pyplot as plt


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
            "image_id": annotation["image_id"],
            "question": question
        }


def analyze_dataset(dataset):
    """
    Analyze the dataset to calculate statistics, such as the number of 'unanswerable' answers.
    Args:
        dataset (Dataset): The VizWiz dataset.
    """
    total_annotations = len(dataset.annotations)
    unanswerable_count = sum(1 for ann in dataset.annotations if ann["answer"] == "unanswerable")
    valid_answers_count = total_annotations - unanswerable_count

    print("\nDataset Statistics:")
    print(f"- Total annotations: {total_annotations}")
    print(f"- Valid answers: {valid_answers_count}")
    print(f"- 'Unanswerable' answers: {unanswerable_count}")
    print(f"- Percentage of 'unanswerable' answers: {unanswerable_count / total_annotations * 100:.2f}%")


def inspect_random_samples(dataset, num_samples=5):
    """
    Inspect random samples from the dataset.
    Args:
        dataset (Dataset): The VizWiz dataset.
        num_samples (int): Number of random samples to inspect.
    """
    print("\nRandom Samples:")
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        print(f"\nSample {i + 1}:")
        print(f"- Image ID: {sample['image_id']}")
        print(f"- Question: {sample['question']}")
        print(f"- Answer: {sample['answer']}")


def visualize_samples(dataset, num_samples=5):
    """
    Visualize random samples from the dataset.
    Args:
        dataset (Dataset): The VizWiz dataset.
        num_samples (int): Number of random samples to visualize.
    """
    print("\nVisualizing Samples:")
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]

        # Load image
        image_path = os.path.join(dataset.image_dir, sample["image_id"])
        image = Image.open(image_path).convert("RGB")

        # Display image with question and answer
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Q: {sample['question']}\nA: {sample['answer']}", fontsize=10)
        plt.show()


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

    # Load the dataset
    dataset = VizWizDataset(annotations_file, image_dir)

    # Analyze the dataset
    analyze_dataset(dataset)

    # Inspect random samples
    inspect_random_samples(dataset)

    # Visualize random samples
    visualize_samples(dataset)

    # Create DataLoader
    dataloader = get_vizwiz_dataloader(annotations_file, image_dir)

    # Iterate through batches
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i + 1}:")
        print(f"- Image batch size: {batch['image'].shape}")
        print(f"- Tokenized question shape: {batch['input_ids'].shape}")
        print(f"- Sample answers: {batch['answer'][0]}")
        if i >= 4:  # Process up to 5 batches
            break


if __name__ == "__main__":
    main()
