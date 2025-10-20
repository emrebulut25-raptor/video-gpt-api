from fastapi import FastAPI

app = FastAPI(
    title="Video GPT API",
    description="Extracts scene-by-scene text prompts and emotions from uploaded videos.",
    version="1.0.0",
    servers=[
        {"url": "https://video-gpt-api.onrender.com"}  # 👈 Tek bir servers satırı olacak
    ]
)

@app.get("/")
def home():
    return {"message": "🚀 FastAPI çalışıyor!"}


from fastapi import UploadFile, File

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    # Burada video analizi yapılacak
    # Şimdilik test için sadece dosya adını döndürelim
    return {"message": f"Video '{file.filename}' analiz edildi!"}

from fastapi import UploadFile, File
import cv2
import tempfile
import os

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    # 1️⃣ Dosyayı geçici klasöre kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    # 2️⃣ Video'yu OpenCV ile aç
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        return {"error": "Video açılamadı."}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    scenes = []
    prev_frame = None
    scene_start = 0

    # 3️⃣ Kare kare ilerle
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Kareler arası fark — sahne geçişini tespit et
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            non_zero = cv2.countNonZero(diff)
            if non_zero > 500000:  # bu eşik: sahne değişimi
                scene_end = i / fps
                scenes.append({
                    "scene_start": round(scene_start, 2),
                    "scene_end": round(scene_end, 2),
                    "description": f"Scene with visual change around {round(scene_end, 2)} seconds"
                })
                scene_start = scene_end
        prev_frame = gray

    cap.release()
    os.remove(temp_path)

    return {
        "total_duration": round(duration, 2),
        "scene_count": len(scenes),
        "scenes": scenes
    }

