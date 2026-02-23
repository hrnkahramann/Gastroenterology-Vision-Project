# Derin Öğrenme ile Gastroenteroloji Görüntü Sınıflandırması

Bu proje, gastroenteroloji alanındaki medikal görüntülerin derin öğrenme yöntemleri kullanılarak sınıflandırılmasını amaçlamaktadır. Çalışma kapsamında hem özel tasarlanmış Convolutional Neural Network (CNN) mimarileri hem de transfer öğrenme yaklaşımları kullanılmıştır.

---

## 📌 Proje Amacı

Gastrointestinal görüntüler üzerinde sınıflandırma modelleri geliştirerek farklı mimarilerin performansını analiz etmek.

Bu kapsamda:

- Özel bir CNN modeli tasarlanmış ve eğitilmiştir.
- Transfer Learning yaklaşımı ile MobileNetV2 mimarisi kullanılmıştır.
- Model performansları doğruluk (accuracy) ve kayıp (loss) metrikleri üzerinden değerlendirilmiştir.
- Eğitim süreci ve sonuçlar karşılaştırmalı olarak analiz edilmiştir.

---

## 📂 Proje Yapısı

Gastroenterology-Vision-Project/
│
├── notebooks/
│   ├── cnn_experiments.ipynb
│   ├── mobilenetv2_experiments.ipynb
│
├── src/
│   ├── train_clip.py
│   ├── evaluate_clip.py
│   └── inference.py
│
├── requirements.txt
├── .gitignore
└── README.md

---

## 📊 Veri Seti

Kullanılan veri seti: **HyperKvasir (Kaggle)**

Veri seti boyut ve lisans kısıtlamaları nedeniyle bu repoya dahil edilmemiştir.

İndirme bağlantısı:  
https://www.kaggle.com/datasets/andrewmvd/hyper-kvasir

Veri seti indirildikten sonra proje dizini içinde aşağıdaki klasöre yerleştirilmelidir:

data/

---

## 🔧 Kullanım

### 1️⃣ Veri Hazırlama

- Veri seti uygun klasör yapısına yerleştirilir.
- Gerekli ön işleme adımları notebook içerisinde uygulanır.

### 2️⃣ Model Eğitimi

- `cnn_experiments.ipynb` dosyası ile özel CNN modeli eğitilir.
- `mobilenetv2_experiments.ipynb` dosyası ile transfer learning uygulanır.
- Model parametreleri ihtiyaçlara göre düzenlenebilir.

### 3️⃣ Model Değerlendirme

- Eğitim sürecinde doğruluk ve kayıp grafikleri incelenir.
- Performans karşılaştırmaları yapılır.
- Gerekirse hiperparametre optimizasyonu uygulanır.

---

## ⚙️ Kurulum

python -m venv .venv  
.venv\Scripts\activate  
pip install -r requirements.txt  

---

## 🧠 Kullanılan Teknolojiler

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 💻 Gereksinimler

- Python 3.8+
- GPU desteği (önerilir)
- En az 8GB RAM (veri boyutuna bağlı olarak)

---

## 📌 Notlar

- Notebook dosyaları Jupyter ortamında çalıştırılacak şekilde hazırlanmıştır.
- Model ve veri yolu (path) sisteminize göre düzenlenmelidir.
- Bu çalışma araştırma ve deneysel model geliştirme amacıyla hazırlanmıştır.
