

!pip install torch torchvision transformers numpy scikit-learn matplotlib pandas

import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms

from google.colab import drive
drive.mount('/content/drive')

# Configuration
CONFIG = {
    "model_name": "openai/clip-vit-base-patch32",
    "batch_size": 64,  # Increased batch size for better performance
    "num_epochs": 10,
    "learning_rate": 5e-5,
    "train_json": "train_data.json",
    "val_json": "val_data.json",
    "test_json": "test_data.json",
    "base_path": "/content/drive/....",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Eksik dosyaları atlayan dataset sınıfı
class HyperKvasirDataset(Dataset):
    def __init__(self, data, transform):
        self.data = []
        self.labels = []
        for item in tqdm(data, desc="Loading Dataset"):
            image_path = item['image_path']
            if not os.path.exists(image_path):
                print(f"Skipping missing file: {image_path}")
                continue
            self.data.append(item)
            self.labels.append(item['caption'])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        label = item['caption']

        image = Image.open(image_path).convert("RGB")
        processed = self.transform(images=image, return_tensors="pt")
        return processed['pixel_values'].squeeze(0), label

# Verileri yükleme
train_data = load_and_fix_paths(CONFIG["train_json"])
val_data = load_and_fix_paths(CONFIG["val_json"])
test_data = load_and_fix_paths(CONFIG["test_json"])

# CLIP işlemleri ve veri yükleyiciler
processor = CLIPProcessor.from_pretrained(CONFIG["model_name"])
train_dataset = HyperKvasirDataset(train_data, processor)
val_dataset = HyperKvasirDataset(val_data, processor)
test_dataset = HyperKvasirDataset(test_data, processor)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

# Model ve optimizasyon
model = CLIPModel.from_pretrained(CONFIG["model_name"]).to(CONFIG["device"])
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

# Etiket haritalaması
label_to_idx = {label: idx for idx, label in enumerate(sorted({d['caption'] for d in train_data}))}
idx_to_label = {v: k for k, v in label_to_idx.items()}

Model kaydetme ve yükleme fonksiyonları
def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="model.pth"):
    model.load_state_dict(torch.load(path, map_location=CONFIG["device"]))
    print(f"Model loaded from {path}")
    return model

# Eğitimde metrikleri kaydetmek için global değişkenler
train_losses = []
train_accuracies = []

# Eğitim fonksiyonu
def train_model():
    model.train()
    for epoch in range(CONFIG["num_epochs"]):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}", mininterval=10):
            images = images.to(CONFIG["device"])
            labels = torch.tensor([label_to_idx[l] for l in labels]).to(CONFIG["device"])

            # Metin özelliklerini alma
            text_inputs = processor.tokenizer(list(label_to_idx.keys()), return_tensors="pt", padding=True).to(CONFIG["device"])
            text_features = model.get_text_features(**text_inputs)

            optimizer.zero_grad()
            # Görsel özellikler ve sınıflandırma
            image_features = model.get_image_features(images)
            logits = torch.matmul(image_features, text_features.T)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Değerlendirme fonksiyonu
def evaluate_model(loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", mininterval=10):
            images = images.to(CONFIG["device"])
            labels_idx = [label_to_idx[l] for l in labels]

            outputs = model.get_image_features(images)
            logits = torch.matmul(outputs, model.get_text_features(**processor.tokenizer(list(label_to_idx.keys()), return_tensors="pt", padding=True).to(CONFIG["device"])).T)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_idx)

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_to_idx.keys()))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# Metrikleri çizdirme fonksiyonu
def plot_metrics():
    epochs = range(1, CONFIG["num_epochs"] + 1)

    # Loss grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Accuracy grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# İlk 5 görsel ve tahminlerini görselleştirme
def visualize_predictions(loader):
    model.eval()
    data_iter = iter(loader)
    images, labels = next(data_iter)

    with torch.no_grad():
        images = images[:5].to(CONFIG["device"])
        outputs = model.get_image_features(images)
        text_features = model.get_text_features(**processor.tokenizer(list(label_to_idx.keys()), return_tensors="pt", padding=True).to(CONFIG["device"]))
        logits = torch.matmul(outputs, text_features.T)
        preds = logits.argmax(dim=1).cpu().numpy()

    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"True: {labels[i]}\nPred: {idx_to_label[preds[i]]}")
        plt.axis("off")
    plt.show()

# Çalıştırma
if __name__ == "__main__":
    print("Starting Training...")
    train_model()
    save_model(model, path="/content/.....")
    plot_metrics()
    print("Validation Results:")
    evaluate_model(val_loader)
    print("Test Results:")
    evaluate_model(test_loader)
    visualize_predictions(test_loader)