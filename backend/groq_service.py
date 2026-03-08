import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_emergency_summary(disaster, confidence, locations, severity="Unknown"):
    prompt = f"""
A multimodal AI crisis detection system analysed a social media post:

Disaster Type: {disaster}
Confidence: {confidence:.4f} ({confidence*100:.1f}%)
Severity Level: {severity}
Locations Detected: {locations}

Generate a professional emergency response report:
1. SITUATION OVERVIEW (1-2 sentences)
2. IMMEDIATE ACTIONS REQUIRED (2-3 bullet points)
3. AFFECTED AREAS (brief note on locations and impact)

Keep under 150 words. Use clear, actionable emergency language.
"""
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert emergency response coordinator with experience in disaster management."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=250
    )
    return chat_completion.choices[0].message.content.strip()


def generate_batch_summary(type_counts: dict, avg_confidence: float, total: int) -> str:
    """Generate a high-level AI summary for a batch of analysed tweets."""
    breakdown = "\n".join([f"  - {k}: {v} tweets" for k, v in type_counts.items()])
    dominant = max(type_counts, key=type_counts.get) if type_counts else "Unknown"

    prompt = f"""
A crisis detection AI has just processed a batch of {total} social media posts.

Results breakdown:
{breakdown}

Average confidence: {avg_confidence*100:.1f}%
Dominant crisis type: {dominant}

Generate a concise batch analysis report (under 160 words) covering:
1. OVERALL SITUATION: What the batch of posts collectively indicates
2. PRIORITY CRISES: Which crisis types need the most urgent attention
3. RECOMMENDED RESPONSE: High-level coordinated response strategy

Use clear, professional emergency management language.
"""
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a senior emergency operations analyst specializing in social media crisis intelligence."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=300
    )
    return chat_completion.choices[0].message.content.strip()