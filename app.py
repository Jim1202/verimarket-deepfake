import streamlit as st
import textstat

RISK_KEYWORDS = [
    "breaking", "urgent", "collapse", "secret", "leaked",
    "crisis", "bankruptcy", "insider", "dump", "panic",
    "liquidity crisis", "imminent crash", "fraud",
    "emergency", "investigation"
]

def financial_risk_score(text):
    text_lower = text.lower()

    keyword_hits = sum([1 for word in RISK_KEYWORDS if word in text_lower])
    sensational_score = keyword_hits / len(RISK_KEYWORDS)

    readability = textstat.flesch_reading_ease(text)
    complexity_score = 1 - max(min(readability / 100, 1), 0)

    ai_score = min((sensational_score * 1.2 + complexity_score * 0.8), 1)

    final_score = (
        0.6 * ai_score +
        0.4 * sensational_score
    )

    return round(final_score, 3), round(sensational_score, 3), round(complexity_score, 3)

def risk_label(score):
    if score > 0.75:
        return "🔴 HIGH RISK"
    elif score > 0.5:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

st.set_page_config(page_title="VeriMarket – Financial Deepfake Detector")

st.title("🔍 VeriMarket – Financial Deepfake Detector")
st.markdown("MVP Risk Scoring Model for Financial Misinformation")

text_input = st.text_area("Paste financial news or announcement here:", height=200)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter text before analyzing.")
    else:
        final_score, sensational_score, complexity_score = financial_risk_score(text_input)

        st.subheader("📊 Risk Assessment")
        st.metric("Manipulation Risk Score", final_score)
        st.write(risk_label(final_score))

        st.markdown("### 🔬 Model Breakdown")
        st.write(f"Sensationalism Score: {sensational_score}")
        st.write(f"Linguistic Complexity Score: {complexity_score}")

st.markdown("---")
st.caption("⚠️ MVP heuristic model. Production version integrates transformer-based AI.")
