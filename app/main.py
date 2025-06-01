from fastapi import FastAPI, UploadFile, File
import shutil
import os
from detect_language import detect_language
from load_cnn_model import CNNClassifier
import torchaudio
import uvicorn
import torch
import torch.nn.functional as F
import torch.nn as nn
app = FastAPI()

TARGET_LEN = 64000
LABELS_MAP = {'en': 0, 'ja': 1, 'vi': 2}
IDX_TO_LANG = {v: k for k, v in LABELS_MAP.items()}

# ---------- DỰ ĐOÁN ----------
def predict_language(model, filepath, device):
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        resample = torchaudio.transforms.Resample(sr, 16000)
        waveform = resample(waveform)
    waveform = (waveform - waveform.mean()) / waveform.std()

    if waveform.shape[1] > TARGET_LEN:
        waveform = waveform[:, :TARGET_LEN]
    else:
        pad_len = TARGET_LEN - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    with torch.no_grad():
        model.eval()
        pred = model(waveform.unsqueeze(0).to(device)).argmax(1).item()
        return IDX_TO_LANG.get(pred, "Unknown")

@app.post("/detect-language/")
async def detect(file: UploadFile = File(...)):
    temp_file = f"temp_audio.wav"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            model = CNNClassifier()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load("language_cnn_model.pkl", map_location="cpu"))
            model.eval()
            model.to(device)  # nếu dùng CUDA	
            lang_cnn = predict_language(model, temp_file, device)
            print("🌐 Ngôn ngữ dự đoán:", lang_cnn)
            lang_whisper,segments = detect_language(temp_file)
            if lang_cnn == lang_whisper:
              model_name = "CNN"
            else:
              model_name = "whisper"  
            language = lang_whisper     
        finally:
            os.remove(temp_file)

        return {"language": language, "text": segments, "model": model_name}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Lấy cổng từ môi trường
    uvicorn.run(app, host="0.0.0.0", port=port)
