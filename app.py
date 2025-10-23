from fastapi import FastAPI, UploadFile, File
import cv2
import tempfile
import os
import numpy as np

app = FastAPI(
    title="Video GPT API",
    description="Extracts scene-by-scene text prompts and emotions from uploaded videos.",
    version="1.0.0",
    servers=[{"url": "https://video-gpt-api-1.onrender.com"}]
)

@app.get("/")
def home():
    return {"message": "🚀 FastAPI çalışıyor!"}

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

def emotion_from_mood(mood: str) -> str:
    if "intense" in mood or "passionate" in mood:
        return "POSITIVE"
    if "calm" in mood or "sad" in mood:
        return "NEGATIVE"
    if "hopeful" in mood or "natural" in mood:
        return "POSITIVE"
    return "NEUTRAL"

def prompt_from_emotion(emotion: str, mood: str) -> str:
    base = f"A {emotion.lower()} cinematic scene, {mood} atmosphere, cinematic lighting"
    if emotion == "POSITIVE":
        return base + ", warm tones, gentle camera movement"
    if emotion == "NEGATIVE":
        return base + ", cool tones, subtle grain, slow zoom"
    return base + ", neutral palette, steady framing"

@app.post("/analyze_video")
async def analyze_video(video: UploadFile = File(..., description="Upload a video file (.mp4, .mov, .avi)")):
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
    diff_threshold = max(150000, int(width * height * 0.25))

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if i > 0:
                scene_end = i / fps
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((scene_start + scene_end) / 2 * fps))
                ok, mid_frame = cap.read()
                if ok:
                    mood = color_mood_from_frame(mid_frame)
                    emotion = emotion_from_mood(mood)
                    prompt = prompt_from_emotion(emotion, mood)
                    story = {
                        "POSITIVE": "The camera captures a moment of joy or relief.",
                        "NEGATIVE": "The scene depicts tension, sadness, or conflict.",
                        "NEUTRAL": "A neutral moment connecting two scenes."
                    }[emotion]
                    scenes.append({
                        "scene_start": round(scene_start, 2),
                        "scene_end": round(scene_end, 2),
                        "emotion": emotion,
                        "prompt": prompt,
                        "storyboard": story
                    })
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
                    story = {
                        "POSITIVE": "The camera captures a moment of joy or relief.",
                        "NEGATIVE": "The scene depicts tension, sadness, or conflict.",
                        "NEUTRAL": "A neutral moment connecting two emotional scenes."
                    }[emotion]
                    scenes.append({
                        "scene_start": round(scene_start, 2),
                        "scene_end": round(scene_end, 2),
                        "emotion": emotion,
                        "prompt": prompt,
                        "storyboard": story
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
