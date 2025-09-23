import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf
import matplotlib.pyplot as plt

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
MODEL_PATH = "notebooks/models/audio_emotion_recognition_model.keras"
SCALER_PATH = "notebooks/models/audio_emotion_recognition_model.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]


# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    # Zero Crossing Rate
    # Extract the zcr here
    # features.extend(zcr)
    zcr = librosa.feature.zero_crossing_rate(data)
    features.extend(np.mean(zcr, axis=1))

    # Chroma STFT
    # Extract the chroma stft here
    # features.extend(chroma)
    chroma = librosa.feature.chroma_stft(y=data, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    # MFCCs
    # Extract the mfccs here
    # features.extend(mfccs)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1))

    # RMS
    # Extract the rms here
    # features.extend(rms)
    rms = librosa.feature.rms(y=data)
    features.extend(np.mean(rms, axis=1))

    # Mel Spectrogram
    # Extract the mel here
    # features.extend(mel)
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    features.extend(np.mean(mel, axis=1))

    # Garantir que tenha exatamente 162 features (ou truncar/zerar)
    target_length = 155
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)
# Code here
st.title("Detector de Emoções em Áudio")
st.write("Envie um arquivo de áudio para detectar a emoção dele.")

# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_audio_path, format="audio/wav")
    
    features = extract_features(temp_audio_path)
    features_scaled = scaler.transform(features)

    preds = model.predict(features_scaled)
    pred_label = EMOTIONS[np.argmax(preds)]

    st.subheader("Emoção detectada:")
    st.write(f"**{pred_label}**")

    st.subheader("Probabilidades por emoção:")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#2B2B2B")
    ax.set_facecolor("#2B2B2B")

    bars = ax.bar(EMOTIONS, preds[0], color="#1f77b4")

    # Texto e ticks claros
    ax.set_title("Distribuição das Emoções", color="white", fontsize=16)
    ax.set_ylabel("Probabilidade", color="white")
    ax.set_xlabel("Emoção", color="white")
    ax.tick_params(colors="white", rotation=45)
    
    # Grid discreto
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#555555")
    st.pyplot(fig)

    os.remove(temp_audio_path)