import streamlit as st

# -----------------------------
# Risk Keywords
# -----------------------------
RISK_KEYWORDS = [
    "breaking", "urgent", "collapse", "secret", "leaked",
    "crisis", "bankruptcy", "insider", "dump", "panic",
    "liquidity crisis", "imminent crash", "fraud",
    "emergency", "investigation", "scandal", "exposed"
]

# -----------------------------
# Risk Scoring Engine (MVP)
# -----------------------------
def financial_risk_score(text):
    text_lower = text.lower()

    # Keyword-based anomaly
    keyword_hits = sum(1 for word in RISK_KEYWORDS if word in text_lower)
    keyword_score = keyword_hits / len(RISK_KEYWORDS)

    # Length anomaly (very short or very long suspicious)
    length_score = min(len(text) / 800, 1)

    # Capitalization anomaly (fake urgency signal)
    capital_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    # Final weighted score
    final_score = (
        0.5 * keyword_score +
        0.3 * length_score +
        0.2 * capital_ratio
    )

    return round(min(final_score, 1), 3), keyword_score, capital_ratio

# -----------------------------
# Risk Label
# -----------------------------
def risk_label(score):
    if score > 0.75:
        return "🔴 HIGH RISK – Escalation Recommended"
    elif score > 0.5:
        return "🟠 MEDIUM RISK – Manual Review Suggested"
    else:
        return "🟢 LOW RISK – No Immediate Concern"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="VeriMarket – Financial Deepfake Detector", layout="centered")

st.title("🔍 VeriMarket – Financial Deepfake Detector")
st.markdown(
    """
    MVP Risk Scoring Engine for detecting potential financial misinformation.
    
    This prototype simulates AI-driven anomaly detection for:
    - Market manipulation signals
    - Sensational announcements
    - Structural anomalies
    """
)

text_input = st.text_area(
    "Paste financial news or announcement here:",
    height=200
)

if st.button("Analyze"):

    if text_input.strip() == "":
        st.warning("Please enter text before analyzing.")
    else:
        score, keyword_score, capital_ratio = financial_risk_score(text_input)

        st.subheader("📊 Risk Assessment")
        st.metric("Manipulation Risk Score", score)
        st.write(risk_label(score))

        st.markdown("### 🔬 Risk Indicators")
        st.write(f"Keyword Anomaly Score: {round(keyword_score,3)}")
        st.write(f"Capitalization Urgency Signal: {round(capital_ratio,3)}")

        if score > 0.75:
            st.error("⚠️ Escalate to Human Validator + Record on Blockchain (Full Version)")
        elif score > 0.5:
            st.warning("Review recommended before publishing or trading decision.")
        else:
            st.success("Content appears structurally normal.")

st.markdown("---")
st.caption(
    "⚠️ MVP heuristic model. Production architecture integrates transformer-based AI, oracle monitoring, and blockchain anchoring."
)
