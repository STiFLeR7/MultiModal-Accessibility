import os
import json

# Paths
ANNOTATION_PATH = "D:/MultiModal-Accessibility/datasets/VizWiz-2023/Annotations/train.json"
IMAGE_DIR = "D:/MultiModal-Accessibility/datasets/VizWiz-2023/train/train"

def load_annotations(annotation_path):
    """Load annotations from JSON file."""
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} annotations.")
        return annotations
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {annotation_path}.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file. {e}")
        return []

def check_missing_images(annotations, image_dir):
    """Check for missing images in the dataset."""
    missing_images = []

    def find_image(image_name, image_dir):
        """Find the image in the directory, handling case differences."""
        for file in os.listdir(image_dir):
            if file.lower() == image_name.lower():
                return os.path.join(image_dir, file)
        return None

    for annotation in annotations:
        # Normalize path and handle case sensitivity
        image_path = find_image(annotation["image"], image_dir)
        if not image_path:
            print(f"Missing: {annotation['image']}")
            missing_images.append(annotation["image"])
        else:
            print(f"Found: {image_path}")
    
    print(f"Total missing images: {len(missing_images)}")
    if missing_images:
        print("Sample Missing Image:", missing_images[0])
    return missing_images

def filter_valid_annotations(annotations):
    """Filter out annotations with null or invalid answers."""
    filtered_annotations = []
    for annotation in annotations:
        valid_answers = [
            ans["answer"] for ans in annotation.get("answers", [])
            if ans.get("answer_confidence") == "yes"
        ]
        if valid_answers:
            filtered_annotations.append({
                "image": annotation["image"],
                "question": annotation["question"],
                "answers": valid_answers
            })
    print(f"Total annotations with valid answers: {len(filtered_annotations)}")
    return filtered_annotations

def debug_annotations(annotations):
    """Print a few annotations for debugging."""
    print("Sample Annotations for Debugging:")
    for idx, annotation in enumerate(annotations[:5]):  # Display first 5 annotations
        print(f"Annotation {idx + 1}:")
        print(json.dumps(annotation, indent=2))

def main():
    # Step 1: Load Annotations
    annotations = load_annotations(ANNOTATION_PATH)
    if not annotations:
        return

    # Step 2: Debug Loaded Annotations
    print("Debugging Loaded Annotations...")
    debug_annotations(annotations)

    # Step 3: Check for Missing Images
    print("\nChecking for Missing Images...")
    check_missing_images(annotations, IMAGE_DIR)

    # Step 4: Filter Valid Annotations
    print("\nFiltering Valid Annotations...")
    valid_annotations = filter_valid_annotations(annotations)

    # Step 5: Debug Filtered Annotations
    print("\nDebugging Filtered Annotations...")
    debug_annotations(valid_annotations)

if __name__ == "__main__":
    main()
