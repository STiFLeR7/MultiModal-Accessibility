import os
import json
import shutil
from tqdm import tqdm

# Paths
DATASET_DIR = "D:/MultiModal-Accessibility/datasets/VizWiz-2023"
ANNOTATION_DIR = os.path.join(DATASET_DIR, "Annotations")
PREPROCESSED_DIR = "D:/MultiModal-Accessibility/preprocessed/VizWiz-2023"

# Ensure output directories exist
os.makedirs(os.path.join(PREPROCESSED_DIR, "annotations"), exist_ok=True)
os.makedirs(os.path.join(PREPROCESSED_DIR, "images/train"), exist_ok=True)
os.makedirs(os.path.join(PREPROCESSED_DIR, "images/val"), exist_ok=True)
os.makedirs(os.path.join(PREPROCESSED_DIR, "images/test"), exist_ok=True)


def preprocess_annotations(annotation_file, output_file):
    """Preprocess annotations and save them to a file."""
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    for item in tqdm(data, desc="Processing annotations"):
        # Adjust keys based on actual dataset structure
        processed_data.append({
            "image_id": item.get("image", item.get("image_id")),
            "question": item.get("question", ""),
            "answer": item.get("answer", None),
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4)


def preprocess_images(source_folder, dest_folder):
    """Copy images to the preprocessed folder."""
    for root, _, files in os.walk(source_folder):
        for file in tqdm(files, desc=f"Processing images in {source_folder}"):
            if file.endswith((".jpg", ".jpeg", ".png")):  # Check for image files
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_folder, file)
                shutil.copy(src_path, dest_path)  # Copy the file


# Process each dataset split
splits = ["train", "val", "test"]
for split in splits:
    annotation_file = os.path.join(ANNOTATION_DIR, f"{split}.json")
    output_annotation_file = os.path.join(PREPROCESSED_DIR, "annotations", f"{split}.json")
    preprocess_annotations(annotation_file, output_annotation_file)

    source_image_folder = os.path.join(DATASET_DIR, split)
    dest_image_folder = os.path.join(PREPROCESSED_DIR, "images", split)
    preprocess_images(source_image_folder, dest_image_folder)

print("Preprocessing completed successfully!")
