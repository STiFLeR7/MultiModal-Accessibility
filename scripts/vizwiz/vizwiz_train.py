import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import torchvision.transforms as T
from torchvision.models import resnet50
from tqdm import tqdm
from PIL import Image
import json
import random
from transformers import CLIPModel, CLIPProcessor


def build_answer_vocab(annotations_file):
    """
    Build a vocabulary of answers from the dataset.
    Args:
        annotations_file (str): Path to the annotations JSON file.
    Returns:
        dict: A mapping from answers to integer labels.
    """
    with open(annotations_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    vocab = set()
    for ann in annotations:
        vocab.add(ann["answer"])

    answer_vocab = {answer: idx for idx, answer in enumerate(sorted(vocab))}
    print(f"Answer vocabulary size: {len(answer_vocab)}")
    return answer_vocab


class VizWizDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, image_dir, tokenizer_model="openai/clip-vit-base-patch32", max_length=77, answer_vocab=None):
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
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

        # Balance between valid answers and "unanswerable"
        valid = [ann for ann in self.annotations if ann["answer"] != "unanswerable"]
        unanswerable = [ann for ann in self.annotations if ann["answer"] == "unanswerable"]
        self.annotations = valid + random.sample(unanswerable, min(len(valid), len(unanswerable)))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = os.path.join(self.image_dir, annotation["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        question = annotation["question"]
        tokenized_question = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,  # Set to 77 for CLIP
            return_tensors="pt",
        )

        # Convert answer to class label
        answer = self.answer_vocab.get(annotation["answer"], -1)  # -1 for unknown answers
        return {
            "image": image,
            "input_ids": tokenized_question["input_ids"].squeeze(0),
            "attention_mask": tokenized_question["attention_mask"].squeeze(0),
            "answer": answer,
        }



class CLIPVQAModel(nn.Module):
    def __init__(self, num_classes):
        super(CLIPVQAModel, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, 512)

        # Projection layers to align feature dimensions
        self.text_projector = nn.Linear(self.clip_model.config.text_config.hidden_size, 512)
        self.image_projector = nn.Identity()  # ResNet-50 already outputs 512

        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 512),  # Combine text and image embeddings
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, images, input_ids, attention_mask):
        # Encode text using CLIP
        text_features = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_features = self.text_projector(text_features)

        # Encode images using ResNet-50
        image_features = self.image_encoder(images)
        image_features = self.image_projector(image_features)

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

        scheduler.step()  # Adjust learning rate
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
    epochs = 5

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
    model = CLIPVQAModel(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, train_dataloader, optimizer, criterion, scheduler, device, epochs)

    # Evaluate the model
    evaluate(model, val_dataloader, criterion, device)


if __name__ == "__main__":
    main()
