from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from model import predict, label_encoder
from groq_service import generate_emergency_summary, generate_batch_summary
import shutil, os, io
from PIL import Image

app = FastAPI(title="CrisisVision AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Severity helper ─────────────────────────────────────────────────
def get_severity_level(confidence: float) -> str:
    if confidence >= 0.85: return "Critical"
    elif confidence >= 0.70: return "High"
    elif confidence >= 0.50: return "Medium"
    else: return "Low"

def make_neutral_image_bytes() -> bytes:
    """Create a plain grey 224x224 JPEG for text-only predictions."""
    img = Image.new("RGB", (224, 224), color=(80, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": "MultiModal BERT + ResNet50",
        "version": "2.0",
        "ner_model": "xlm-roberta-base-ner-hrl"
    }

@app.get("/classes")
async def get_classes():
    classes = label_encoder.classes_.tolist()
    return {"total_classes": len(classes), "disaster_types": classes}


@app.post("/predict/")
async def get_prediction(text: str = Form(...), image: UploadFile = File(...)):
    image_path = f"temp_{image.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    result = predict(text, image_path)
    os.remove(image_path)

    severity = get_severity_level(result["confidence"])
    result["severity_level"] = severity

    summary = generate_emergency_summary(
        disaster=result["disaster_type"],
        confidence=result["confidence"],
        locations=result["locations_detected"],
        severity=severity
    )
    result["ai_summary"] = summary
    return result


@app.post("/predict-batch/")
async def predict_batch(
    texts: str = Form(...),          # JSON-encoded list of strings
    image: UploadFile = File(None)   # optional representative image
):
    """Analyse multiple tweet texts at once. Returns per-tweet results + aggregate stats."""
    import json
    tweet_list = json.loads(texts)

    # Prepare image bytes (shared representative or neutral grey)
    if image and image.filename:
        img_bytes = await image.read()
    else:
        img_bytes = make_neutral_image_bytes()

    results = []
    for i, tweet in enumerate(tweet_list):
        tmp_path = f"batch_temp_{i}.jpg"
        with open(tmp_path, "wb") as f:
            f.write(img_bytes)
        try:
            r = predict(tweet, tmp_path)
            r["severity_level"] = get_severity_level(r["confidence"])
            r["tweet"] = tweet
            results.append(r)
        except Exception as e:
            results.append({"tweet": tweet, "error": str(e)})
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Aggregate stats
    valid = [r for r in results if "disaster_type" in r]
    from collections import Counter
    type_counts = Counter(r["disaster_type"] for r in valid)
    avg_confidence = sum(r["confidence"] for r in valid) / len(valid) if valid else 0
    severity_counts = Counter(r["severity_level"] for r in valid)

    # Batch AI summary
    batch_summary = generate_batch_summary(type_counts, avg_confidence, len(tweet_list))

    return {
        "total": len(tweet_list),
        "analysed": len(valid),
        "results": results,
        "aggregate": {
            "disaster_type_counts": dict(type_counts),
            "severity_counts": dict(severity_counts),
            "avg_confidence": round(avg_confidence, 4),
            "dominant_type": type_counts.most_common(1)[0][0] if type_counts else "Unknown"
        },
        "ai_batch_summary": batch_summary
    }
