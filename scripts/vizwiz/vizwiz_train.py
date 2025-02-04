import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as T
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from tqdm import tqdm
from PIL import Image
import json
import random


def build_answer_vocab(annotations_file):
    with open(annotations_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    vocab = set()
    for ann in annotations:
        vocab.add(ann["answer"])

    answer_vocab = {answer: idx for idx, answer in enumerate(sorted(vocab))}
    print(f"Answer vocabulary size: {len(answer_vocab)}")
    return answer_vocab


class VizWizDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, image_dir, max_length=77, answer_vocab=None):
        self.image_dir = image_dir
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.max_length = max_length
        self.answer_vocab = answer_vocab

        self.image_transform = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
        ])

        with open(annotations_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        # Balance "unanswerable" answers instead of removing too many
        valid = [ann for ann in self.annotations if ann["answer"] != "unanswerable"]
        unanswerable = [ann for ann in self.annotations if ann["answer"] == "unanswerable"]
        self.annotations = valid + random.sample(unanswerable, min(len(valid) * 3 // 4, len(unanswerable)))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = os.path.join(self.image_dir, annotation["image_id"])
        image = Image.open(image_path).convert("RGB")

        question = annotation["question"]
        tokenized_question = self.processor.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,  # CLIP requires max_length=77
            return_tensors="pt",
        )

        image = self.image_transform(image)

        # Convert answer to class label
        answer = self.answer_vocab.get(annotation["answer"], -1)  # -1 for unknown answers
        return {
            "image": image,
            "input_ids": tokenized_question["input_ids"].squeeze(0),
            "attention_mask": tokenized_question["attention_mask"].squeeze(0),
            "answer": answer,
        }


class VQAModel(nn.Module):
    def __init__(self, num_classes):
        super(VQAModel, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # Use EfficientNet-B3 as the image encoder
        self.image_encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.image_encoder.classifier = nn.Linear(self.image_encoder.classifier[1].in_features, 512)

        # Classifier with increased dropout to prevent overfitting
        self.classifier = nn.Sequential(
            nn.Linear(512 + self.clip_model.config.projection_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout from 0.3 to 0.5
            nn.Linear(512, num_classes),
        )

    def forward(self, images, input_ids, attention_mask):
        # CLIP text encoding
        text_features = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        # EfficientNet image encoding
        image_features = self.image_encoder(images)

        # Combine image and text features
        combined_features = torch.cat((text_features, image_features), dim=1)
        logits = self.classifier(combined_features)
        return logits


def train(model, dataloader, optimizer, criterion, scheduler, device, epochs, checkpoint_path="vizwiz_checkpoint.pth"):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"].to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, answers)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step(running_loss)  # Adjust learning rate if validation loss plateaus
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")

        # Save checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"].to(device)

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, answers)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == answers).sum().item()
            total += answers.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def main():
    annotations_file = "D:/MultiModal-Accessibility/preprocessed/VizWiz-2023/annotations/train.json"
    image_dir = "D:/MultiModal-Accessibility/preprocessed/VizWiz-2023/images/train"
    batch_size = 16
    learning_rate = 1e-5
    epochs = 15  # Keeping at 15 for better convergence

    # Build answer vocabulary
    answer_vocab = build_answer_vocab(annotations_file)
    num_classes = len(answer_vocab)

    # Dataset and DataLoader
    dataset = VizWizDataset(annotations_file, image_dir, answer_vocab=answer_vocab)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, optimizer, scheduler, and loss function
    model = VQAModel(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, train_dataloader, optimizer, criterion, scheduler, device, epochs)

    # Evaluate the model
    evaluate(model, val_dataloader, criterion, device)


if __name__ == "__main__":
    main()
