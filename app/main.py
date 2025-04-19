from fastapi import FastAPI, UploadFile, File
import shutil
import os
from detect_language import detect_language

app = FastAPI()

@app.post("/detect-language/")
async def detect(file: UploadFile = File(...)):
    temp_file = f"temp_audio.wav"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            language = detect_language(temp_file)
        finally:
            os.remove(temp_file)

        return {"language": language}
    except Exception as e:
        return {"error": str(e)}
