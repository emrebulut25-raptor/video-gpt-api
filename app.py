from fastapi import FastAPI, UploadFile, File, Form
import cv2
import tempfile
import os
import numpy as np

app = FastAPI(
    title="Video GPT API",
    description="Extracts multilingual scene-by-scene emotional analysis from uploaded videos.",
    version="2.0.0",
    servers=[{"url": "https://video-gpt-api-1.onrender.com"}]
)

@app.get("/")
def home():
    return {"message": "🚀 Multilingual Video Emotion Analyzer is running!"}

# 🧩 Renk tabanlı duygu tespiti
def color_mood_from_frame(frame: np.ndarray) -> str:
    mean_color = np.mean(frame, axis=(0, 1))
    b, g, r = mean_color
    if r > g and r > b:
        return "passionate or intense"
    elif b > r and b > g:
        return "calm or melancholic"
    elif g > r and g > b:
        return "hopeful or natural"
    else:
        return "balanced or neutral"

# 🧠 Renk moduna göre duygu tahmini
def emotion_from_mood(mood: str) -> str:
    if "intense" in mood or "passionate" in mood:
        return "POSITIVE"
    if "melancholic" in mood:
        return "NEGATIVE"
    if "hopeful" in mood:
        return "POSITIVE"
    return "NEUTRAL"

# 🎬 Dinamik prompt üretimi (çeşitlilik)
def prompt_from_emotion(emotion: str, mood: str) -> str:
    base = f"A {emotion.lower()} cinematic scene with {mood} atmosphere."
    if emotion == "POSITIVE":
        options = [
            "warm lighting, gentle camera pans, uplifting tone",
            "soft sunlight, vivid colors, dynamic framing",
            "joyful pace, smiling characters, immersive movement"
        ]
    elif emotion == "NEGATIVE":
        options = [
            "cold tones, slow zoom, emotional tension",
            "dim lighting, subtle shadows, heavy silence",
            "muted palette, slow cuts, reflective expression"
        ]
    else:
        options = [
            "steady framing, balanced colors, neutral pacing",
            "transitional moment, connecting two emotions",
            "documentary tone, realistic light balance"
        ]
    return base + " " + np.random.choice(options)

# 🌍 Çok dilli çeviri
def translate_output(text: str, lang: str) -> str:
    translations = {
        "tr": {"POSITIVE": "POZİTİF", "NEGATIVE": "NEGATİF", "NEUTRAL": "NÖTR"},
        "es": {"POSITIVE": "POSITIVO", "NEGATIVE": "NEGATIVO", "NEUTRAL": "NEUTRO"},
        "zh": {"POSITIVE": "积极", "NEGATIVE": "消极", "NEUTRAL": "中性"},
        "en": {"POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL"},
    }
    return translations.get(lang, translations["en"]).get(text, text)

# 🎞️ Video analizi
@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(..., description="Upload a video file (.mp4, .mov, .avi)"),
    lang: str = Form("tr")
):
    contents = await video.read()
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
    diff_threshold = max(150000, int(width * height * 0.5))  # 🔧 burada 0.5 yaptık

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
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((scene_start + scene_end) / 2 * fps))
                ok, mid_frame = cap.read()
                if ok:
                    mood = color_mood_from_frame(mid_frame)
                    emotion = emotion_from_mood(mood)
                    prompt = prompt_from_emotion(emotion, mood)
                    scenes.append({
                        "scene_start": round(scene_start, 2),
                        "scene_end": round(scene_end, 2),
                        "emotion": translate_output(emotion, lang),
                        "prompt": prompt,
                        "mood": mood
                    })
                scene_start = scene_end
        prev_gray = gray
        i += 1

    cap.release()
    os.remove(temp_path)

    return {
        "total_duration": round(duration, 2),
        "scene_count": len(scenes),
        "language": lang,
        "scenes": scenes
    }
