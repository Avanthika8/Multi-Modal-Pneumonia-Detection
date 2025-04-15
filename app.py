import gradio as gr
import numpy as np
import tensorflow as tf
import librosa
from PIL import Image
import tensorflow_hub as hub

# Load models
image_model = tf.keras.models.load_model("EffNetB0-Pneumonia-96_38.h5")
audio_model = tf.keras.models.load_model("multiclass_audio_model.h5")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Constants
IMG_SIZE = (256, 256)
SR = 16000
AUDIO_DURATION = 4
class_labels = ["Asthma", "Bronchial", "COPD", "Healthy", "Pneumonia"]
pneumonia_index = class_labels.index("Pneumonia")

# Preprocessing functions
def preprocess_image(img):
    img = img.convert("RGB").resize((224, 224))  # Convert to 3-channel RGB
    img_array = np.array(img).astype("float32") / 255.0
    return img_array.reshape(1, 224, 224, 3)


def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=SR)
    y = y[:SR * AUDIO_DURATION]
    y = np.pad(y, (0, max(0, SR * AUDIO_DURATION - len(y))), mode='constant')
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(waveform)
    embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    return np.expand_dims(embedding, axis=0)

# Main analysis function
def analyze(xray, audio):
    try:
        image_input = preprocess_image(xray)
        audio_input = preprocess_audio(audio)

        image_pred = image_model.predict(image_input)[0][0]
        audio_pred_vector = audio_model.predict(audio_input)[0]
        audio_pred = audio_pred_vector[pneumonia_index]
        fused_score = 0.3 * image_pred + 0.7 * audio_pred

        label = "Pneumonia" if fused_score >= 0.45 else "Normal"
        return (
            f"Diagnosis: {label}",
            f"X-ray Score: {image_pred:.2f}",
            f"Audio Score: {audio_pred:.2f}",
            f"Fused Score: {fused_score:.2f}"
        )
    except Exception as e:
        return ("Error: " + str(e), "", "", "")

# Gradio UI
demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(label="Chest X-ray (JPG, PNG)", type="pil"),
        gr.Audio(label="Breathing Audio (WAV, MP3)", type="filepath")
    ],
    outputs=[
        gr.Text(label="Prediction"),
        gr.Text(label="X-ray Model Score"),
        gr.Text(label="Audio Model Score"),
        gr.Text(label="Fused Score")
    ],
    title="Pneumonia Detection",
    description="Upload a chest X-ray and a breathing audio clip to detect pneumonia using deep learning."
)

demo.launch()
