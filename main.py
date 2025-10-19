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
