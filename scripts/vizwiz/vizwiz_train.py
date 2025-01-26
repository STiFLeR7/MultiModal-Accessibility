import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
import json


class VizWizDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, image_dir, tokenizer_model="bert-base-uncased", max_length=128):
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_length = max_length
        self.image_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        with open(annotations_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

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
            max_length=self.max_length,
            return_tensors="pt",
        )

        answer = annotation["answer"]
        return {
            "image": image,
            "input_ids": tokenized_question["input_ids"].squeeze(0),
            "attention_mask": tokenized_question["attention_mask"].squeeze(0),
            "answer": answer,
        }


class SimpleVQAModel(nn.Module):
    def __init__(self, text_model_name="bert-base-uncased", num_classes=512):
        super(SimpleVQAModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = models.resnet18(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, 512)

        self.classifier = nn.Sequential(
            nn.Linear(512 + self.text_encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, images, input_ids, attention_mask):
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        image_features = self.image_encoder(images)
        combined_features = torch.cat((text_features, image_features), dim=1)
        logits = self.classifier(combined_features)
        return logits


def train(model, dataloader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"]

            # Dummy target (you'll need to preprocess answers for classification tasks)
            targets = torch.zeros(len(answers), dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")


def main():
    annotations_file = "D:/MultiModal-Accessibility/preprocessed/VizWiz-2023/annotations/train.json"
    image_dir = "D:/MultiModal-Accessibility/preprocessed/VizWiz-2023/images/train"
    batch_size = 16
    learning_rate = 1e-4
    epochs = 5
    num_classes = 512  # Adjust based on your task

    # Dataset and DataLoader
    dataset = VizWizDataset(annotations_file, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, optimizer, and loss function
    model = SimpleVQAModel(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # Adjust based on task

    # Train the model
    train(model, dataloader, optimizer, criterion, device, epochs)


if __name__ == "__main__":
    main()
