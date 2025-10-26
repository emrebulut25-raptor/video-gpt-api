from fastapi import FastAPI, UploadFile, File, Form
import cv2, tempfile, os, numpy as np

app = FastAPI(
    title="Video GPT API",
    description="Extracts scene-by-scene text prompts and emotions from uploaded videos.",
    version="1.1.0",
    servers=[{"url": "https://video-gpt-api-1.onrender.com"}]
)

@app.get("/")
def home():
    return {"message": "🚀 Multilingual Video Emotion Analyzer is running!"}


@app.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(..., description="Upload a video file (.mp4, .mov, .avi)"),
    lang: str = Form("en")  # 👈 frontend’den gelen dil bilgisi
):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        os.remove(temp_path)
        return {"error": "Video açılamadı."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    scenes = []
    prev_gray = None
    scene_start = 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    diff_threshold = max(150000, int(width * height * 0.25))
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            non_zero = cv2.countNonZero(diff)
            if non_zero > diff_threshold:
                scene_end = i / fps
                mood = "passionate" if np.mean(frame[:,:,2]) > 100 else "calm"
                emotion = "POSITIVE" if "passionate" in mood else "NEGATIVE"
                prompt = f"A {emotion.lower()} cinematic scene, {mood} atmosphere"
                scenes.append({
                    "scene_start": round(scene_start, 2),
                    "scene_end": round(scene_end, 2),
                    "emotion": emotion,
                    "mood": mood,
                    "prompt": prompt
                })
                scene_start = scene_end
        prev_gray = gray
        i += 1

    cap.release()
    os.remove(temp_path)

    # 🌍 Dil desteği mesajları
    translations = {
        "en": "Video emotion analysis completed successfully!",
        "tr": "Video duygu analizi başarıyla tamamlandı!",
        "es": "¡Análisis de emociones del video completado con éxito!",
        "zh": "视频情感分析已成功完成！"
    }

    return {
        "language": lang,
        "message": translations.get(lang, translations["en"]),
        "total_duration": round(duration, 2),
        "scene_count": len(scenes),
        "scenes": scenes
    }
