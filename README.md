# 🧠 Multi-Modal Pneumonia Detection

A deep learning system that combines **chest X-ray** and **breathing audio** inputs to accurately detect **pneumonia** using both image and audio modalities.

🔗 **Try it live on Hugging Face Spaces:**  
👉 [Multi-modal Pneumonia Detection Demo](https://huggingface.co/spaces/AvanthikaKatari/Multi-modal_Pneumonia_Detection)

---

## 🚀 Project Overview

This project leverages the power of deep learning by combining two diagnostic sources:

- **Chest X-ray images** processed via an EfficientNetB0 model.
- **Respiratory audio recordings** analyzed using YAMNet and a custom multiclass classifier.

The results from both modalities are fused using a weighted scoring mechanism to improve diagnostic accuracy, particularly for pneumonia detection.

---

## 🧩 Key Features

- ✅ Multi-modal deep learning: Image + Audio
- 📦 Gradio UI for real-time user interaction
- 🔍 CLAHE and filtering for image enhancement
- 🎧 YAMNet embeddings for audio processing
- ⚡ Fused prediction score improves model reliability
- 🖥️ Deployable on Raspberry Pi for edge inference
- 🌐 Deployed and accessible via Hugging Face Spaces

---

## 🛠️ Tech Stack

- **Languages:** Python
- **Libraries:** TensorFlow, TensorFlow Hub, NumPy, Librosa, PIL, Gradio
- **Models Used:** 
  - EfficientNetB0 for X-ray classification
  - YAMNet for audio embedding
  - Custom CNN for audio classification
- **Deployment:** Hugging Face Spaces, Gradio, Raspberry PI 4
