from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from model import predict
from groq_service import generate_emergency_summary
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def get_prediction(
    text: str = Form(...),
    image: UploadFile = File(...)
):

    image_path = f"temp_{image.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    result = predict(text, image_path)

    os.remove(image_path)

    # 🔥 Call Groq GenAI
    summary = generate_emergency_summary(
        disaster=result["disaster_type"],
        confidence=result["confidence"],
        locations=result["locations_detected"]
    )

    result["ai_summary"] = summary

    return result