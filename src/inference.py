

!pip install flask transformers tqdm pillow matplotlib scikit-learn torch

import os
import threading
import json
import requests
import torch
from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Google Drive'ı Bağlama
from google.colab import drive
drive.mount('/content/drive')

# Ana Path (Drive Üzerindeki Klasör)
BASE_PATH = "/content/driv....."

# Flask Uygulaması
app = Flask(__name__)

# Model ve İşlemci Yükleme
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

# Etiket Haritası Oluşturma
# JSON dosyalarından etiketleri çıkarmak
def extract_labels(json_paths):
    labels = set()
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
        for item in data:
            labels.add(item["caption"])
    return sorted(labels)

# JSON dosyalarının yolları
json_files = [
    os.path.join(BASE_PATH, "train_data.json"),
    os.path.join(BASE_PATH, "val_data.json"),
    os.path.join(BASE_PATH, "test_data.json")
]

# Etiket haritasını oluştur
labels = extract_labels(json_files)
label_to_idx = {label: idx for idx, label in enumerate(labels)}
idx_to_label = {v: k for k, v in label_to_idx.items()}

print("Label to Index Mapping:", label_to_idx)

# Görüntüyü İşleme Fonksiyonu
def process_image(image):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    return inputs['pixel_values']

# Flask API Tahmin Rota
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Tahmin İşlemi
        image = Image.open(file).convert("RGB")
        inputs = process_image(image)

        with torch.no_grad():
            image_features = model.get_image_features(inputs)
            text_inputs = processor.tokenizer(
                list(label_to_idx.keys()), return_tensors="pt", padding=True
            ).to(DEVICE)
            text_features = model.get_text_features(**text_inputs)
            logits = torch.matmul(image_features, text_features.T)
            pred_idx = logits.argmax(dim=1).item()
            predicted_label = idx_to_label[pred_idx]

        return jsonify({"predicted_label": predicted_label}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask Server'ı Arkaplanda Çalıştırma
def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# Test Verilerini Yükleme
def load_test_data(json_file):
    json_path = os.path.join(BASE_PATH, json_file)
    with open(json_path, 'r') as f:
        return json.load(f)

# İlk 10 Görseli ve Tahminlerini Gösterme
def visualize_samples(data, predicted_labels=None):
    plt.figure(figsize=(20, 10))
    for i in range(10):
        if i >= len(data):
            break
        item = data[i]
        image_path = os.path.join(BASE_PATH, item['image_path'])
        image = Image.open(image_path).convert("RGB")

        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        title = f"True: {item['caption']}"
        if predicted_labels:
            title += f"\nPred: {predicted_labels[i]}"
        plt.title(title)
        plt.axis("off")
    plt.show()

# Doğruluk Testi ve Metrik Çizimleri
def evaluate_rag(test_data):
    true_labels = []
    predicted_labels = []

    for item in tqdm(test_data, desc="Evaluating RAG"):
        image_path = os.path.join(BASE_PATH, item['image_path'])
        true_label = item['caption']
        try:
            response = requests.post(
                "http://0.0.0.0:5000/predict",
                files={"file": open(image_path, "rb")}
            )
            predicted_label = response.json().get("predicted_label", "unknown")
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Görsellerin İlk 10'unu Görselleştir
    visualize_samples(test_data, predicted_labels)

    # Doğruluk Metrikleri
    print("Classification Report:")
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    print(report)
    print("Confusion Matrix:")
    matrix = confusion_matrix(true_labels, predicted_labels)
    print(matrix)

    # Grafiksel Gösterimler
    plot_metrics(report, matrix)

    # Sonuçları Kaydetme
    report_path = os.path.join(BASE_PATH, "classification_report.json")
    with open(report_path, 'w') as f:
        json.dump({
            "classification_report": report,
            "confusion_matrix": matrix.tolist()
        }, f, indent=4)
    print(f"Results saved to {report_path}")

# Metrikleri Çizdirme
def plot_metrics(report, matrix):
    # F1-Skorları Bar Grafiği
    plt.figure(figsize=(10, 5))
    classes = list(report.keys())[:-3]  # Son 3 metrik "accuracy", "macro avg", "weighted avg"
    f1_scores = [report[cls]["f1-score"] for cls in classes]
    plt.bar(classes, f1_scores, color='skyblue')
    plt.title("F1 Scores by Class")
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45)
    plt.show()

    # Karışıklık Matrisi
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

# Ana Çalıştırma Bloğu
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["flask", "evaluate"], default="flask", help="Run Flask API or Evaluate Model")
    parser.add_argument("--test_json", type=str, default="test_data.json", help="Path to test data JSON (relative to BASE_PATH)")
    args = parser.parse_args()

    if args.mode == "flask":
        print("Starting Flask API...")
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.setDaemon(True)  # Arka planda çalıştır
        flask_thread.start()
        print("Flask server started on http://0.0.0.0:5000")
    elif args.mode == "evaluate":
        print("Starting evaluation...")
        test_data = load_test_data(args.test_json)
        visualize_samples(test_data)  # İlk 10 veriyi göster
        evaluate_rag(test_data)