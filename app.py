import streamlit as st
import torch
import textstat
from transformers import pipeline

# -----------------------------
# Load Lightweight Model
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# -----------------------------
# Financial Risk Keywords
# -----------------------------
RISK_KEYWORDS = [
    "breaking", "urgent", "collapse", "secret", "leaked",
    "crisis", "bankruptcy", "insider", "dump", "panic",
    "liquidity crisis", "imminent crash", "fraud",
    "emergency", "investigation"
]

# -----------------------------
# Risk Scoring Function
# -----------------------------
def financial_risk_score(text):
    text_lower = text.lower()

    # Keyword detection score
    keyword_hits = sum([1 for word in RISK_KEYWORDS if word in text_lower])
    sensational_score = keyword_hits / len(RISK_KEYWORDS)

    # Linguistic complexity score
    readability = textstat.flesch_reading_ease(text)
    complexity_score = 1 - max(min(readability / 100, 1), 0)

    # AI classification (sentiment proxy)
    result = classifier(text)[0]

    if result["label"] == "NEGATIVE":
        ai_score = result["score"]
    else:
        ai_score = 1 - result["score"]

    # Final weighted risk score
    final_score = (
        0.5 * ai_score +
        0.3 * sensational_score +
        0.2 * complexity_score
    )

    return (
        round(final_score, 3),
        round(ai_score, 3),
        round(sensational_score, 3),
        round(complexity_score, 3)
    )

# -----------------------------
# Risk Label
# -----------------------------
def risk_label(score):
    if score > 0.75:
        return "🔴 HIGH RISK"
    elif score > 0.5:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="VeriMarket – Financial Deepfake Detector",
    layout="centered"
)

st.title("🔍 VeriMarket – Financial Deepfake Detector")
st.markdown(
    "Detect potential AI-generated or manipulated financial news "
    "using multi-factor risk scoring."
)

text_input = st.text_area(
    "Paste financial news or announcement here:",
    height=200
)

if st.button("Analyze"):

    if text_input.strip() == "":
        st.warning("Please enter text before analyzing.")
    else:
        with st.spinner("Analyzing..."):
            final_score, ai_score, sensational_score, complexity_score = financial_risk_score(text_input)

        st.subheader("📊 Risk Assessment")
        st.metric("Final Manipulation Risk Score", final_score)
        st.write(risk_label(final_score))

        st.markdown("### 🔬 Model Breakdown")
        st.write(f"AI Sentiment Risk Score: {ai_score}")
        st.write(f"Sensationalism Score: {sensational_score}")
        st.write(f"Linguistic Complexity Score: {complexity_score}")

        if final_score > 0.75:
            st.error("Escalation Recommended: Human Expert Review + Blockchain Anchoring")
        elif final_score > 0.5:
            st.warning("Manual review suggested.")
        else:
            st.success("Low probability of manipulation.")

st.markdown("---")
st.caption(
    "⚠️ MVP Model: Lightweight transformer used for demonstration purposes. "
    "Detection is probabilistic and not deterministic."
)
