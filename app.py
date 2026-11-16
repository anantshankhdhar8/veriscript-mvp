import streamlit as st
import re
import statistics
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="VeriScript AI Detector", page_icon="🛡️", layout="centered")

# --- STAGE II: LOAD TRANSFORMER ---
@st.cache_resource
def load_transformer():
    return pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")

heavy_lifter = load_transformer()

# --- STAGE I: HEURISTICS ---
def calculate_heuristics(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if len(sentences) < 2:
        return {"burstiness": 0.0, "ttr": 0.0, "stage1_score": 0.5}
        
    sentence_lengths = [len(s.split()) for s in sentences]
    mean_len = statistics.mean(sentence_lengths)
    std_dev = statistics.stdev(sentence_lengths)
    
    if std_dev + mean_len == 0:
        burstiness = 0.0
    else:
        burstiness = (std_dev - mean_len) / (std_dev + mean_len)
        
    words = text.lower().split()
    unique_words = set(words)
    ttr = len(unique_words) / len(words) if len(words) > 0 else 0.0
    
    ai_prob = (abs(burstiness) - 0.2) * 1.5
    if ttr < 0.5:
        ai_prob += 0.2
        
    ai_prob = max(0.01, min(0.99, ai_prob))
    
    return {"burstiness": round(burstiness, 3), "ttr": round(ttr, 3), "stage1_score": round(ai_prob, 2)}

# --- UI STOREFRONT ---
st.title("🛡️ VeriScript")
st.subheader("Hybrid Heuristic-Transformer AI Text Detection")

with st.expander("🔍 How VeriScript Works (Cascade Architecture)"):
    st.markdown("""
    1. **Stage I (Fast Path):** Analyzes statistical *Burstiness* (sentence variation) and *Type-Token Ratio* (vocabulary richness).
    2. **Confidence Gate:** If Stage I is highly confident (≤0.20 or ≥0.80), it returns a verdict instantly.
    3. **Stage II (Escalation):** Ambiguous borderline scores are escalated to a Deep Learning Transformer (RoBERTa) for context analysis.
    """)

user_text = st.text_area("Paste text to analyze (minimum 2 sentences):", height=200)

if st.button("Run Detection", type="primary"):
    if len(user_text.split()) < 10:
        st.warning("Please paste a longer text block for accurate analysis.")
    else:
        st.divider()
        st.info("⏱️ Running Stage I: Heuristic Analysis (CPU)...")
        
        metrics = calculate_heuristics(user_text)
        score = metrics['stage1_score']
        
        # Display Dashboard
        col1, col2, col3 = st.columns(3)
        col1.metric("Burstiness (Variance)", metrics["burstiness"])
        col2.metric("Type-Token Ratio", metrics["ttr"])
        col3.metric("Stage I AI Probability", f"{score}")
        
        st.progress(score, text="Stage I Confidence Distribution (0=Human, 1=AI)")
        
        # --- STAGE III: TRAFFIC COP LOGIC ---
        if score <= 0.20:
            st.success(f"✅ **Verdict: Fast Path Confident Human** (Score: {score})")
        elif score >= 0.80:
            st.error(f"🚨 **Verdict: Fast Path Confident AI** (Score: {score})")
        else:
            st.warning(f"⚠️ **Borderline Score ({score}). Escalating to Stage II Transformer...**")
            
            with st.spinner("🧠 Stage II Transformer analyzing deep context (GPU/Heavy Compute)..."):
                result = heavy_lifter(user_text)[0]
                label = result['label']
                confidence = result['score']
                
                st.divider()
                if label == "ChatGPT":
                    st.error(f"🚨 **Stage II Final Verdict: AI-Generated** (Model Confidence: {confidence:.2%})")
                else:
                    st.success(f"✅ **Stage II Final Verdict: Human-Written** (Model Confidence: {confidence:.2%})")