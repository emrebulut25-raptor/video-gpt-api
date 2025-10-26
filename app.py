from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import tempfile
import os
import numpy as np

app = FastAPI(
    title="Video GPT API",
    description="Multilingual Video Emotion Analyzer",
    version="2.0.0"
)

# Statik dosyalar için klasör oluştur
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import tempfile
import os
import numpy as np

app = FastAPI()

# index.html dosyasını ana sayfa olarak göster
@app.get("/", response_class=HTMLResponse)
def home():
    file_path = os.path.join(os.getcwd(), "index.html")
    if not os.path.exists(file_path):
        return "<h2 style='color:red;'>⚠️ index.html bulunamadı!</h2>"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Basit test endpoint
@app.get("/status")
def status():
    return {"message": "✅ API çalışıyor!"}


# Renkten ruh hali çıkarımı
def color_mood_from_frame(frame: np.ndarray) -> str:
    mean_color = np.mean(frame, axis=(0, 1))
    b, g, r = mean_color
    if r > g and r > b:
        return "passionate or intense"
    elif b > r and b > g:
        return "calm or sad"
    elif g > r and g > b:
        return "natural or hopeful"
    else:
        return "balanced or neutral"

# Duygu tahmini
def emotion_from_mood(mood: str) -> str:
    if "intense" in mood or "passionate" in mood:
        return "POSITIVE"
    if "calm" in mood or "sad" in mood:
        return "NEGATIVE"
    if "hopeful" in mood or "natural" in mood:
        return "POSITIVE"
    return "NEUTRAL"

# Çok dilli prompt oluşturma
def prompt_from_emotion(emotion: str, mood: str, lang: str) -> str:
    prompts = {
        "en": {
            "POSITIVE": f"A positive cinematic scene, {mood} atmosphere, warm tones, gentle camera movement.",
            "NEGATIVE": f"A melancholic or tense cinematic scene, {mood} tone, cool lighting, slow zoom.",
            "NEUTRAL": f"A balanced neutral cinematic moment, {mood} tone, soft natural light."
        },
        "tr": {
            "POSITIVE": f"Pozitif bir sinematik sahne, {mood} atmosfer, sıcak tonlar, yumuşak kamera hareketi.",
            "NEGATIVE": f"Hüzünlü veya gergin bir sinematik sahne, {mood} tonlar, soğuk ışık, yavaş yakınlaşma.",
            "NEUTRAL": f"Dengeli, nötr bir sinematik sahne, {mood} atmosfer, doğal ışık."
        },
        "es": {
            "POSITIVE": f"Una escena cinematográfica positiva, atmósfera {mood}, tonos cálidos, movimiento suave de cámara.",
            "NEGATIVE": f"Una escena tensa o melancólica, tono {mood}, iluminación fría, zoom lento.",
            "NEUTRAL": f"Una escena cinematográfica neutral, atmósfera {mood}, luz natural y equilibrio."
        },
        "zh": {
            "POSITIVE": f"积极的电影场景，{mood}氛围，暖色调，柔和的镜头移动。",
            "NEGATIVE": f"忧郁或紧张的电影场景，{mood}色调，冷光，缓慢的变焦。",
            "NEUTRAL": f"平衡中性的电影场景，{mood}氛围，自然光，稳定镜头。"
        }
    }
    return prompts.get(lang, prompts["en"])[emotion]

# 🎥 Video Analizi
@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(..., description="Upload a video file (.mp4, .mov, .avi)"),
    lang: str = Form("en")
):
    contents = await video.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        os.remove(temp_path)
        return {"error": "Video could not be opened."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    scenes = []
    prev_gray = None
    scene_start = 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    diff_threshold = max(50000, int(width * height * 0.05))  # hassasiyet = %5

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
                    prompt = prompt_from_emotion(emotion, mood, lang)
                    scenes.append({
                        "scene_start": round(scene_start, 2),
                        "scene_end": round(scene_end, 2),
                        "emotion": emotion,
                        "prompt": prompt
                    })
                scene_start = scene_end
        prev_gray = gray
        i += 1

    cap.release()
    os.remove(temp_path)

    return {
        "total_duration": round(duration, 2),
        "scene_count": len(scenes),
        "scenes": scenes
    }

# Başlangıç mesajı
@app.get("/status")
def status():
    return {"message": "🚀 Multilingual Video Emotion Analyzer is running!"}
