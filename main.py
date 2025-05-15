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
            language,segments = detect_language(temp_file)
        finally:
            os.remove(temp_file)

        return {"language": language, "text": segments}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Lấy cổng từ môi trường
    uvicorn.run(app, host="0.0.0.0", port=port)