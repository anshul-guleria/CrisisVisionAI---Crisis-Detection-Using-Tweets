import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_emergency_summary(disaster, confidence, locations):

    prompt = f"""
    A disaster detection AI system has produced the following result:

    Disaster Type: {disaster}
    Confidence Score: {confidence}
    Locations Detected: {locations}

    Generate a professional emergency response summary in 3-4 sentences.
    Focus on urgency, impact, and suggested response actions.
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an emergency response assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=200
    )

    return chat_completion.choices[0].message.content.strip()