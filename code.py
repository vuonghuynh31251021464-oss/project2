import streamlit as st
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="HybridOracle", layout="wide", initial_sidebar_state="expanded")

# ====================== ULTRA GENZ BACKGROUND ======================
st.markdown("""
<style>
    /* Background chính siêu đẹp */
    body {
        background: linear-gradient(135deg, #0a0022, #1a0033, #2a1a4d);
        background-attachment: fixed;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Overlay gradient + glow */
    .main {
        background: rgba(20, 10, 50, 0.75);
        backdrop-filter: blur(16px);
        border-radius: 28px;
        padding: 2.8rem;
        border: 1px solid rgba(0, 255, 255, 0.3);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.6),
                    inset 0 0 40px rgba(0, 255, 255, 0.1);
        margin-top: 1rem;
    }
    
    h1 {
        font-size: 3.8rem;
        background: linear-gradient(90deg, #ff00cc, #00ffff, #ffff00, #ff00cc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-shadow: 0 0 40px rgba(255, 0, 204, 0.6);
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px #ff00cc; }
        to { text-shadow: 0 0 40px #00ffff; }
    }
    
    h2, h3 {
        color: #00ffff;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #ff00cc, #00ffff, #ff00cc);
        background-size: 200% 200%;
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        border-radius: 50px;
        padding: 1rem 3rem;
        border: none;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.7);
        transition: all 0.4s;
        animation: pulse 2s infinite;
    }
    
    .stButton>button:hover {
        transform: scale(1.1) translateY(-3px);
        box-shadow: 0 0 50px rgba(255, 0, 204, 0.9);
    }
    
    .result-box {
        background: linear-gradient(135deg, rgba(255,0,204,0.25), rgba(0,255,255,0.25));
        padding: 2.5rem;
        border-radius: 25px;
        border: 2px solid rgba(0,255,255,0.5);
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.3);
    }
    
    .stSlider, .stSelectbox {
        background: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 HYBRIDORACLE")
st.markdown("### 🎧 **Music Vibe Detector 3.0** ✨")
st.markdown("**AI đoán gu nhạc GenZ chỉ trong tích tắc** 🔥")

# Sidebar
with st.sidebar:
    st.markdown("## ⚡ VIBE CHECK ACTIVE")
    st.markdown("**Music Taste AI v3.0**")
    st.info("• Perceptron Model\n• 24 Training Samples\n• GenZ Neon Edition")
    st.markdown("---")
    st.caption("Made with neon & love for GenZ")

# ----------------------------
# Utility Function
# ----------------------------
def train_perceptron(X, y):
    model = Perceptron(max_iter=2000, tol=1e-3, random_state=42)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)
    return model, scaler

# ======================== MAIN CONTENT ========================
st.header("🎧 What's Your Music Vibe Today?")

# Input Layout
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    age = st.slider("🧍 Age", 13, 35, 22)

with c2:
    hours = st.slider("🎧 Hours / Day", 0, 15, 4)

with c3:
    habit = st.selectbox("📱 Main Platform", ["Spotify", "YouTube", "TikTok", "Radio"])
    habit_code = {"Radio":0, "Spotify":1, "YouTube":2, "TikTok":3}[habit]

# Dataset
X = np.array([
    [20,4,1], [30,2,0], [25,6,2], [40,1,0], [18,8,2], [35,3,1],
    [22,5,2], [28,3,1], [33,7,2], [45,2,0], [19,9,2], [38,4,1],
    [24,6,2], [29,2,0], [26,5,2], [42,1,1], [21,7,2], [31,3,0],
    [27,8,2], [36,2,1], [23,6,2], [39,4,0], [20,5,2], [34,3,1]
])

y = np.array([2,1,2,1,2,0, 2,1,2,1,2,0, 2,1,2,1,2,0, 2,1,2,1,2,0])

if st.button("🔥 GUESS MY VIBE NOW", type="primary", use_container_width=True):
    
    model, scaler = train_perceptron(X, y)
    input_data = np.array([[age, hours, habit_code if habit_code < 3 else 2]])
    pred = model.predict(scaler.transform(input_data))[0]
    
    genres = ["🎤 Pop", "🎸 Rock", "🔥 EDM"]
    result = genres[pred]
    
    st.markdown(f"""
    <div class="result-box">
        <h2>YOUR VIBE IS</h2>
        <h1 style="font-size: 4.2rem; margin: 10px 0;">{result}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    accuracy = model.score(scaler.transform(X), y) * 100
    st.metric("Model Accuracy", f"{accuracy:.1f}%")
    
    st.markdown("### 🎯 Your Music Personality")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Age**: {age} | **Listening**: {hours} giờ/ngày")
    with col_b:
        st.write(f"**Platform**: {habit}")
    
    if pred == 2:
        st.success("**EDM OVERLOAD!** 🔥\nBạn sống cho những đêm rave, bass mạnh và drop điên cuồng!")
    elif pred == 1:
        st.warning("**ROCK REBEL** 🎸\nBạn có cá tính mạnh, thích cảm xúc và không ngại nổi loạn!")
    else:
        st.info("**POP VIBES** ✨\nBạn chill, yêu trend, dễ thương và luôn cập nhật hit mới!")

    st.balloons()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #00ffff; font-size: 1.1rem;'>
    🔮 <strong>HybridOracle</strong> — Music Vibe Detector 3.0<br>
    <span style='color: #ff00cc;'>Neon Edition • Powered by Perceptron</span>
</div>
""", unsafe_allow_html=True)
