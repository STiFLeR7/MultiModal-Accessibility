import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
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

def load_data(split: str) -> Tuple[List[Dict], List[Image.Image]]:
    """
    Load both annotations and images for a given split.
    Args:
        split (str): Dataset split ('train', 'val', 'test').
    Returns:
        Tuple[List[Dict], List[Image.Image]]: Annotations and images.
    """
    # Load annotations
    annotations = load_annotations(split)

    # Load images
    images = []
    for ann in annotations:
        img_path = os.path.join(IMAGE_DIR, split, ann["image_id"])
        if os.path.exists(img_path):
            images.append(load_image(img_path))
        else:
            print(f"Warning: Image not found at {img_path}")
    
    return annotations, images

def main():
    # Example: Load train split
    split = "train"
    print(f"Loading {split} split...")
    
    annotations, images = load_data(split)
    print(f"Loaded {len(annotations)} annotations and {len(images)} images.")
    
    # Print a sample annotation
    if annotations:
        print("\nSample Annotation:")
        print(json.dumps(annotations[0], indent=2))
    
    # Show a sample image
    if images:
        print("\nDisplaying a sample image...")
        images[0].show()

if __name__ == "__main__":
    main()
