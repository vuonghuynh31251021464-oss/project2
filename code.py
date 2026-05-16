import streamlit as st
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="HybridOracle", layout="wide", initial_sidebar_state="expanded")

# ====================== PASTEL PROFESSIONAL STYLE ======================
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #ffe4f3, #e0f7fa, #f3e0ff, #fff0e6);
        background-attachment: fixed;
        color: #2c2c2c;
    }
    
    .main {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 3rem;
        border: 2px solid rgba(255, 105, 180, 0.3);
        box-shadow: 0 15px 50px rgba(255, 182, 193, 0.25);
    }
    
    h1 {
        font-size: 3.8rem;
        background: linear-gradient(90deg, #ff69b4, #00ced1, #ba55d3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #d63384;
    }
    
    /* Input Fields - Bỏ nền xám, tăng độ tương phản */
    .stSlider, .stSelectbox {
        background: white;
        padding: 15px;
        border-radius: 20px;
        border: 2px solid #ffb6c1;
        box-shadow: 0 4px 15px rgba(255, 182, 193, 0.2);
    }
    
    label {
        color: #2c2c2c !important;
        font-weight: 600;
    }
    
    /* Button chuyên nghiệp */
    .stButton>button {
        background: linear-gradient(45deg, #ff69b4, #00ced1, #ba55d3);
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        border-radius: 50px;
        padding: 1.1rem 3.5rem;
        border: none;
        box-shadow: 0 10px 30px rgba(255, 105, 180, 0.4);
        transition: all 0.4s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 40px rgba(255, 105, 180, 0.6);
    }
    
    /* Result Box */
    .result-box {
        background: linear-gradient(135deg, #fff0f5, #e0f7fa);
        padding: 2.8rem;
        border-radius: 25px;
        border: 3px solid #ff69b4;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 35px rgba(255, 182, 193, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 HYBRIDORACLE")
st.markdown("### 🎧 **Music Vibe Detector** 🌸")
st.markdown("**AI đoán gu nhạc pastel cho GenZ** ✨")

# Sidebar
with st.sidebar:
    st.markdown("## 🌷 Pastel Vibe AI")
    st.info("Perceptron Model • 24 Samples • Pastel Professional Edition")
    st.caption("Soft • Sweet • Accurate 💕")

# Utility Function
def train_perceptron(X, y):
    model = Perceptron(max_iter=2000, tol=1e-3, random_state=42)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)
    return model, scaler

# ======================== INPUT ========================
st.header("🎧 What's Your Music Vibe Today?")

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    age = st.slider("🧍 Your Age", 13, 35, 22)

with c2:
    hours = st.slider("🎧 Hours listening / day", 0, 15, 4)

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

if st.button("🌸 GUESS MY VIBE NOW", type="primary", use_container_width=True):
    
    model, scaler = train_perceptron(X, y)
    input_data = np.array([[age, hours, habit_code if habit_code < 3 else 2]])
    pred = model.predict(scaler.transform(input_data))[0]
    
    genres = ["🎤 Pop", "🎸 Rock", "🔥 EDM"]
    result = genres[pred]
    
    st.markdown(f"""
    <div class="result-box">
        <h2>YOUR VIBE IS</h2>
        <h1 style="font-size: 4.2rem; margin: 15px 0;">{result}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    accuracy = model.score(scaler.transform(X), y) * 100
    st.metric("Model Accuracy", f"{accuracy:.1f}%")
    
    st.markdown("### 🎀 Your Music Personality")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Age**: {age} | **Listening**: {hours} giờ/ngày")
    with col_b:
        st.write(f"**Platform**: {habit}")
    
    if pred == 2:
        st.success("**EDM Sweetheart** 🔥💖 Bạn năng động và thích party!")
    elif pred == 1:
        st.warning("**Rock Cutie** 🎸🌷 Bạn có cá tính mạnh mẽ!")
    else:
        st.info("**Pop Princess / Prince** ✨💕 Bạn chill và yêu trend!")

    st.balloons()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #d63384; font-size: 1.1rem;'>
    🔮 <strong>HybridOracle</strong> — Pastel Music Vibe Detector<br>
    Soft Pastel Professional Edition • Made for GenZ 💕
</div>
""", unsafe_allow_html=True)
