import streamlit as st
import requests
# from transformers import pipeline

BACKEND_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="Crisis Detection AI", layout="wide")

st.title("🚨 AI-Powered Crisis Detection System")

st.markdown("---")

# ----------------------------
# Optional Local GenAI (No API Key Needed)
# ----------------------------
@st.cache_resource
def load_summarizer():
    # return pipeline("text-generation", model="distilgpt2")
    pass

summarizer = load_summarizer()

def generate_summary(disaster, confidence, locations):
    base_text = f"""
    Disaster Type: {disaster}
    Confidence Score: {confidence}
    Locations: {locations}
    Generate a short emergency response summary:
    """

    response = summarizer(base_text, max_length=80, num_return_sequences=1)
    return response[0]["generated_text"]


# ----------------------------
# UI Input Section
# ----------------------------
tweet_text = st.text_area("📝 Enter Tweet Text")

uploaded_image = st.file_uploader(
    "📷 Upload Disaster Image",
    type=["jpg", "jpeg", "png"]
)

if st.button("🔍 Analyze Crisis"):

    if tweet_text and uploaded_image:

        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing crisis..."):

            response = requests.post(
                BACKEND_URL,
                files={"image": uploaded_image},
                data={"text": tweet_text}
            )

        if response.status_code == 200:
            result = response.json()

            disaster = result["disaster_type"]
            confidence = result["confidence"]
            locations = result["locations_detected"]
            summary = result["ai_summary"]

            st.success("Analysis Complete")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Disaster Type")
                st.markdown(f"### {disaster}")
                st.progress(min(float(confidence), 1.0))
                st.write(f"Confidence: {confidence:.4f}")

            with col2:
                st.subheader("Locations Detected")
                st.write(locations)

            st.markdown("---")

            st.subheader("AI Emergency Summary")
            st.info(summary)

            # ----------------------------
            # Expandable Raw Data
            # ----------------------------
            with st.expander("🔎 View Raw Model Output"):
                st.json(result)

        else:
            st.error("Backend Error")

    else:
        st.warning("Please enter tweet text and upload image")