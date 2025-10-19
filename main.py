from fastapi import FastAPI

app = FastAPI(
    title="Video GPT API",
    description="Extracts scene-by-scene text prompts and emotions from uploaded videos.",
    version="1.0.0",
    servers=[
        {"url": "https://video-gpt-api.onrender.com"}  # 👈 bu satır olmazsa ChatGPT Builder hata verir
    ]
)

@app.get("/")
def home():
    return {"message": "🚀 FastAPI çalışıyor!"}
