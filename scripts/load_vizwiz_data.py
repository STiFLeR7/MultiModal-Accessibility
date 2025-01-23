import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Iterator
from PIL import Image

# Paths to preprocessed data
BASE_DIR = "D:/MultiModal-Accessibility/preprocessed/VizWiz-2023"
ANNOTATION_DIR = os.path.join(BASE_DIR, "annotations")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

def load_annotations(split: str) -> List[Dict]:
    """
    Load annotations for a given data split.
    Args:
        split (str): Dataset split ('train', 'val', 'test').
    Returns:
        List[Dict]: List of annotation dictionaries.
    """
    annotation_file = os.path.join(ANNOTATION_DIR, f"{split}.json")
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    return annotations

def load_image(image_path: str) -> Image.Image:
    """
    Load an image given its path.
    Args:
        image_path (str): Path to the image file.
    Returns:
        PIL.Image.Image: Loaded image object.
    """
    return Image.open(image_path).convert("RGB")

def batch_loader(data: List[Dict], batch_size: int) -> Iterator[List[Dict]]:
    """
    Yield batches of data for efficient processing.
    Args:
        data (List[Dict]): The dataset.
        batch_size (int): Number of items per batch.
    Yields:
        Iterator[List[Dict]]: Batches of data.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def load_data_in_batches(split: str, batch_size: int = 100) -> Iterator[Tuple[List[Dict], List[Image.Image]]]:
    """
    Load annotations and images in batches.
    Args:
        split (str): Dataset split ('train', 'val', 'test').
        batch_size (int): Number of items per batch.
    Returns:
        Iterator[Tuple[List[Dict], List[Image.Image]]]: Batches of annotations and images.
    """
    annotations = load_annotations(split)
    for batch in batch_loader(annotations, batch_size):
        images = []
        for ann in batch:
            img_path = os.path.join(IMAGE_DIR, split, ann["image_id"])
            if os.path.exists(img_path):
                images.append(load_image(img_path))
            else:
                print(f"Warning: Image not found at {img_path}")
        yield batch, images

def main():
    split = "train"
    print(f"Loading {split} split in batches...")
    
    batch_size = 50  # Adjust batch size based on your system's memory
    data_loader = load_data_in_batches(split, batch_size)
    
    for i, (annotations, images) in enumerate(data_loader):
        print(f"\nBatch {i + 1}:")
        print(f"- Loaded {len(annotations)} annotations.")
        print(f"- Loaded {len(images)} images.")
        
        # Print a sample annotation
        print("\nSample Annotation:")
        print(json.dumps(annotations[0], indent=2))
        
        # Show a sample image (optional)
        print("\nDisplaying a sample image...")
        images[0].show()
        break  # Only process the first batch for testing

if __name__ == "__main__":
    main()
