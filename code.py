import streamlit as st
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="HybridOracle", layout="wide", initial_sidebar_state="expanded")

# ====================== GENZ ULTRA STYLE ======================
st.markdown("""
<style>
    /* Background Gradient */
    body {
        background: linear-gradient(135deg, #1a0033, #0f0c29, #302b63);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        border-radius: 25px;
        padding: 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    h1 {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #ff00cc, #00ffcc, #ffff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
        text-shadow: 0 0 30px rgba(255, 0, 204, 0.5);
    }
    h2 {
        color: #00ffff;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff00cc, #00ffff);
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        border-radius: 50px;
        padding: 1rem 3rem;
        border: none;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.6);
        transition: all 0.4s;
    }
    .stButton>button:hover {
        transform: scale(1.08);
        box-shadow: 0 0 40px rgba(255, 0, 204, 0.9);
    }
    .result-box {
        background: linear-gradient(135deg, rgba(255,0,204,0.2), rgba(0,255,255,0.2));
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid rgba(0,255,255,0.4);
        text-align: center;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 HYBRIDORACLE")
st.markdown("### 🎧 **Music Vibe Detector 2.0** ✨")
st.markdown("**AI đoán gu nhạc GenZ siêu chuẩn chỉ trong 1 giây** 🔥")

# Sidebar
with st.sidebar:
    st.markdown("## ⚡ VIBE CHECK")
    st.markdown("**Music Taste AI**")
    st.info("Perceptron Model • 24 training samples • GenZ Edition")
    st.markdown("---")
    st.caption("Made with 🔥 for GenZ")

# ----------------------------
# Utility Function
# ----------------------------
def train_perceptron(X, y):
    model = Perceptron(max_iter=2000, tol=1e-3, random_state=42)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)
    return model, scaler

# ======================== MAIN LAYOUT ========================
st.header("🎧 What's Your Music Vibe Today?")

# Bố cục Input đẹp hơn
input_col1, input_col2, input_col3 = st.columns([1, 1, 1])

with input_col1:
    age = st.slider("🧍 Your Age", 13, 35, 22, help="GenZ focus")

with input_col2:
    hours = st.slider("🎧 Hours listening per day", 0, 15, 4)

with input_col3:
    habit = st.selectbox("📱 Main Platform", ["Spotify", "YouTube", "TikTok", "Radio"])
    habit_code = {"Radio":0, "Spotify":1, "YouTube":2, "TikTok":3}[habit]

# ==================== DATASET 24 SAMPLES ====================
X = np.array([
    [20,4,1], [30,2,0], [25,6,2], [40,1,0], [18,8,2], [35,3,1],
    [22,5,2], [28,3,1], [33,7,2], [45,2,0], [19,9,2], [38,4,1],
    [24,6,2], [29,2,0], [26,5,2], [42,1,1], [21,7,2], [31,3,0],
    [27,8,2], [36,2,1], [23,6,2], [39,4,0], [20,5,2], [34,3,1]
])

y = np.array([2,1,2,1,2,0, 2,1,2,1,2,0, 2,1,2,1,2,0, 2,1,2,1,2,0])

# Predict Button
if st.button("🔥 GUESS MY VIBE NOW", type="primary", use_container_width=True):
    
    model, scaler = train_perceptron(X, y)
    input_data = np.array([[age, hours, habit_code if habit_code < 3 else 2]])  # TikTok → YouTube style
    pred = model.predict(scaler.transform(input_data))[0]
    
    genres = ["🎤 Pop", "🎸 Rock", "🔥 EDM"]
    result = genres[pred]
    
    # Kết quả nổi bật
    st.markdown(f"""
    <div class="result-box">
        <h2>YOUR VIBE IS</h2>
        <h1 style="font-size: 3.8rem; margin: 0;">{result}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    accuracy = model.score(scaler.transform(X), y) * 100
    st.metric("Model Accuracy", f"{accuracy:.1f}%", delta="Training Data")
    
    # Phân tích cá nhân hóa
    st.markdown("### 🎯 Your Music Personality Report")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Age**: {age} tuổi")
        st.write(f"**Daily Listening**: {hours} giờ")
    with col_b:
        st.write(f"**Platform**: {habit}")
    
    if pred == 2:
        st.success("**EDM ENERGY!** 🔥\nBạn là tiktoker party, thích bass mạnh, drop điên cuồng và festival!")
    elif pred == 1:
        st.warning("**ROCK REBEL** 🎸\nBạn có cá tính, thích cảm xúc mạnh và không theo đám đông.")
    else:
        st.info("**POP GIRL / BOY** ✨\nBạn chill, yêu trend, dễ thương và nghe nhạc để vui vẻ.")

    st.balloons()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #00ffff;'>
    🔮 <strong>HybridOracle</strong> — Music Vibe Detector 2.0<br>
    Powered by Perceptron • Designed for GenZ
</div>
""", unsafe_allow_html=True)
