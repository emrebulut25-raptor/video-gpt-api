from fastapi import FastAPI, UploadFile, File
import cv2
import tempfile
import os
import numpy as np
from transformers import pipeline

app = FastAPI(
    title="Video GPT API",
    description="Extracts scene-by-scene text prompts and emotions from uploaded videos.",
    version="1.0.0",
    servers=[{"url": "https://video-gpt-api-1.onrender.com"}]
)

@app.get("/")
def home():
    return {"message": "🚀 FastAPI çalışıyor!"}

# Duygu analiz modeli
emotion_analyzer = pipeline("sentiment-analysis")

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Videoyu sahnelere ayırır, her sahnenin duygusal tonunu analiz eder
    ve sinematik promptlar üretir.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        return {"error": "Video açılamadı."}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    scenes = []
    prev_frame = None
    scene_start = 0

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            non_zero = cv2.countNonZero(diff)
            if non_zero > 500000:
                scene_end = i / fps
                midpoint = int((scene_start + scene_end) / 2 * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint)
                _, mid_frame = cap.read()

                mean_color = np.mean(mid_frame, axis=(0, 1))
                r, g, b = mean_color

                if r > g and r > b:
                    color_mood = "passionate or intense"
                elif b > r and b > g:
                    color_mood = "calm or sad"
                else:
                    color_mood = "balanced or neutral"

                scene_text = f"Scene from {round(scene_start,2)}s to {round(scene_end,2)}s appears {color_mood}."
                emotion = emotion_analyzer(scene_text)[0]

                ai_prompt = f"A {emotion['label'].lower()} cinematic scene, {color_mood} atmosphere, cinematic lighting."

                if emotion['label'] == "POSITIVE":
                    story = "The camera captures a moment of joy or relief."
                elif emotion['label'] == "NEGATIVE":
                    story = "The scene depicts tension, sadness, or conflict."
                else:
                    story = "A neutral moment connecting two emotional scenes."

                scenes.append({
                    "scene_start": round(scene_start, 2),
                    "scene_end": round(scene_end, 2),
                    "emotion": emotion['label'],
                    "confidence": round(emotion['score'], 2),
                    "prompt": ai_prompt,
                    "storyboard": story
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
