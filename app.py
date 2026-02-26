import streamlit as st

RISK_KEYWORDS = [
    "breaking", "urgent", "collapse", "secret", "leaked",
    "crisis", "bankruptcy", "insider", "dump", "panic",
    "liquidity crisis", "imminent crash", "fraud",
    "emergency", "investigation"
]

def financial_risk_score(text):
    text_lower = text.lower()

    # Keyword-based risk
    keyword_hits = sum(1 for word in RISK_KEYWORDS if word in text_lower)
    keyword_score = keyword_hits / len(RISK_KEYWORDS)

    # Text length anomaly
    length_score = min(len(text) / 1000, 1)

    # Capital letters ratio (fake urgency proxy)
    capital_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    final_score = (
        0.5 * keyword_score +
        0.3 * length_score +
        0.2 * capital_ratio
    )

    return round(final_score, 3)

def risk_label(score):
    if score > 0.75:
        return "🔴 HIGH RISK"
    elif score > 0.5:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

st.set_page_config(page_title="VeriMarket – Financial Deepfake Detector")

st.title("🔍 VeriMarket – Financial Deepfake Detector")
st.markdown("MVP Risk Scoring Engine for Financial Misinformation")

text_input = st.text_area("Paste financial news or announcement here:", height=200)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter text before analyzing.")
    else:
        score = financial_risk_score(text_input)

        st.subheader("📊 Risk Assessment")
        st.metric("Manipulation Risk Score", score)
        st.write(risk_label(score))

        st.markdown("### 🔬 Model Factors")
        st.write("- Keyword anomaly detection")
        st.write("- Structural length anomaly")
        st.write("- Urgency capitalization signal")

st.markdown("---")
st.caption("⚠️ MVP heuristic model. Production version integrates transformer-based AI & blockchain anchoring.")
