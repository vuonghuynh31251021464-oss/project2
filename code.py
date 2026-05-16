import streamlit as st
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="HybridOracle", layout="wide")

# ====================== GENZ STYLE CSS ======================
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .main {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    h1 {
        font-size: 3.2rem;
        background: linear-gradient(90deg, #ff00cc, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #00ffff;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff00cc, #00ffff);
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        border: none;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(255, 0, 204, 0.8);
    }
    .stSlider label, .stSelectbox label {
        color: #ffffff;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 HYBRIDORACLE")
st.markdown("### 🎧 **Music Vibe Detector** ✨")
st.markdown("**AI đoán gu nhạc GenZ chỉ trong 1 giây** 🔥")

# Sidebar GenZ Style
st.sidebar.markdown("## ⚡ GENZ MODE ACTIVATED")
st.sidebar.markdown("**Music Taste AI**")
st.sidebar.info("Mô hình Perceptron đang phân tích vibe của bạn...")

# ----------------------------
# Utility Function
# ----------------------------
def train_perceptron(X, y):
    model = Perceptron(max_iter=2000, tol=1e-3, random_state=42)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)
    return model, scaler

# ========================
# MUSIC PREFERENCE MODEL
# ========================
st.header("🎧 What's Your Music Vibe?")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("🧍 Your Age", 10, 70, 22)
    hours = st.slider("🎵 Hours listening / day", 0, 12, 4)

with col2:
    habit = st.selectbox("📱 Main Listening App", ["Spotify", "YouTube", "Radio"])
    habit_code = {"Radio":0, "Spotify":1, "YouTube":2}[habit]

# ==================== 24 MẪU DỮ LIỆU ====================
X = np.array([
    [20,4,1], [30,2,0], [25,6,2], [40,1,0], [18,8,2], [35,3,1],
    [22,5,2], [28,3,1], [33,7,2], [45,2,0], [19,9,2], [38,4,1],
    [24,6,2], [29,2,0], [26,5,2], [42,1,1], [21,7,2], [31,3,0],
    [27,8,2], [36,2,1], [23,6,2], [39,4,0], [20,5,2], [34,3,1]
])

y = np.array([2,1,2,1,2,0, 2,1,2,1,2,0, 2,1,2,1,2,0, 2,1,2,1,2,0])

if st.button("🔥 GUESS MY VIBE NOW", type="primary", use_container_width=True):
    
    model, scaler = train_perceptron(X, y)
    input_data = np.array([[age, hours, habit_code]])
    pred = model.predict(scaler.transform(input_data))[0]
    
    genres = ["🎤 Pop", "🎸 Rock", "🔥 EDM"]
    result = genres[pred]
    
    st.success(f"**YOUR VIBE IS: {result}**")
    
    accuracy = model.score(scaler.transform(X), y) * 100
    st.metric("🔬 Model Accuracy", f"{accuracy:.1f}%")
    
    # Phân tích cá nhân hóa GenZ
    st.markdown("### 🎯 Your Music Personality:")
    st.write(f"**Age**: {age} | **Daily Vibe**: {hours}h | **Platform**: {habit}")
    
    if pred == 2:
        st.info("**EDM Energy Overload!** 🔥\nBạn là party animal, thích nhịp bass mạnh, festival, và sống hết mình!")
    elif pred == 1:
        st.info("**Rock Rebel Detected** 🎸\nBạn có cá tính mạnh, thích cảm xúc sâu và bùng nổ!")
    else:
        st.info("**Pop Princess/Prince** ✨\nBạn chill, dễ thương, thích trend và nhạc dễ nghe!")

    st.balloons()

st.markdown("---")
st.caption("🔮 HybridOracle • Built for GenZ • Powered by Perceptron AI")
