import streamlit as st
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AlgoRhythm", layout="wide", initial_sidebar_state="expanded")

# ====================== ULTRA GENZ NEON STYLE ======================
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #0a0022, #1a0033, #2a1a4d);
        background-attachment: fixed;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
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
        padding: 2.8rem;
        border-radius: 25px;
        border: 2px solid rgba(0,255,255,0.6);
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.4);
    }
    .analysis-card {
        background: rgba(255,255,255,0.08);
        padding: 1.8rem;
        border-radius: 20px;
        border: 1px solid rgba(0,255,255,0.4);
        margin: 1.2rem 0;
    }
    .stSlider, .stSelectbox {
        background: rgba(255,255,255,0.08);
        padding: 12px;
        border-radius: 15px;
        border: 1px solid rgba(0,255,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎧 AlgoRhythm 🎵")
st.markdown("### 🎧 **Music Vibe Detector 3.0** ✨")
st.markdown("**AI đoán gu nhạc GenZ chỉ trong tích tắc** 🔥")

# Sidebar
with st.sidebar:
    st.markdown("## ⚡ VIBE CHECK ACTIVE")
    st.markdown("**Music Taste AI v3.0**")
    st.info("• Perceptron Model\n• 24 Training Samples\n• GenZ Neon Edition")
    st.markdown("---")
    st.caption("Made with neon & love for GenZ")

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
    age = st.slider("🧍 Age", 13, 35, 22)

with c2:
    hours = st.slider("🎧 Hours / Day", 0, 15, 4)

with c3:
    platform_options = {
        "🎵 Spotify": 1,
        "▶️ YouTube": 2,
        "♬ TikTok": 3,
        "📻 Radio": 0
    }
    habit_display = st.selectbox(
        "📱 Main Platform",
        options=list(platform_options.keys())
    )
    habit_code = platform_options[habit_display]

# ====================== DATASET CÂN BẰNG HƠN ======================
X = np.array([
    # Pop (0)
    [22,3,1], [19,2,1], [24,4,1], [21,3,0], [26,2,1], [23,3,1], [20,5,1], [25,2,0], [18,4,1], [27,3,1],
    # Rock (1)
    [28,3,0], [35,4,0], [30,2,0], [32,3,1], [27,5,0], [33,2,0], [29,4,2], [31,3,1], [34,4,0],
    # EDM (2)
    [20,6,3], [19,8,2], [22,7,3], [21,9,2], [23,5,3], [18,8,3], [24,6,2], [20,7,3]
])

y = np.array([0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2])

if st.button("🔥 GUESS MY VIBE NOW", type="primary", use_container_width=True):
    
    model, scaler = train_perceptron(X, y)
    input_data = np.array([[age, hours, habit_code if habit_code < 3 else 2]])
    pred = model.predict(scaler.transform(input_data))[0]
    
    genres = ["🎤 Pop", "🎸 Rock", "🔥 EDM"]
    result = genres[pred]
    
    st.markdown(f"""
    <div class="result-box">
        <h2>YOUR VIBE IS</h2>
        <h1 style="font-size: 4.5rem; margin: 10px 0;">{result}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    accuracy = model.score(scaler.transform(X), y) * 100
    st.metric("Model Accuracy", f"{accuracy:.1f}%")
    
    # ====================== PHÂN TÍCH ======================
    st.markdown("### 🎯 Your Music Personality Analysis")
    
    if pred == 2: # EDM
        st.success("**🔥 EDM OVERLOAD** — Bạn là **Party Animal**!")
        st.markdown(f"""
        <div class="analysis-card">
            <strong>📱 Platform ưa thích:</strong> {habit_display}<br><br>
            <strong>🎵 Đặc điểm:</strong> Thích bass mạnh, drop điên cuồng, festival, nhịp nhanh<br><br>
            <strong>🧠 Tính cách:</strong> Năng động, thích thử thách, sống hết mình với âm nhạc<br><br>
            <strong>⭐ Gợi ý nghệ sĩ:</strong> Alan Walker, The Chainsmokers, Martin Garrix, David Guetta, Skrillex, Illenium, Zedd, Kygo, Avicii
        </div>
        """, unsafe_allow_html=True)
        
    elif pred == 1: # Rock
        st.warning("**🎸 ROCK REBEL** — Bạn là **Emotional Rocker**!")
        st.markdown(f"""
        <div class="analysis-card">
            <strong>📱 Platform ưa thích:</strong> {habit_display}<br><br>
            <strong>🎵 Đặc điểm:</strong> Thích guitar riff mạnh, lời bài hát sâu sắc, cảm xúc dâng trào<br><br>
            <strong>🧠 Tính cách:</strong> Cá tính mạnh, độc lập, đôi khi nổi loạn<br><br>
            <strong>⭐ Gợi ý nghệ sĩ:</strong> Imagine Dragons, Linkin Park, Billie Eilish, Coldplay, Twenty One Pilots, The Weeknd, Arctic Monkeys, Foo Fighters
        </div>
        """, unsafe_allow_html=True)
        
    else: # Pop
        st.info("**✨ POP VIBES** — Bạn là **Trendy Pop Lover**!")
        st.markdown(f"""
        <div class="analysis-card">
            <strong>📱 Platform ưa thích:</strong> {habit_display}<br><br>
            <strong>🎵 Đặc điểm:</strong> Thích nhạc dễ nghe, bắt tai, theo trend, giai điệu vui tươi<br><br>
            <strong>🧠 Tính cách:</strong> Vui vẻ, hòa đồng, yêu sự tươi mới<br><br>
            <strong>⭐ Gợi ý nghệ sĩ:</strong> Taylor Swift, Olivia Rodrigo, Sabrina Carpenter, Ariana Grande, NewJeans, BLACKPINK, BTS, Charlie Puth, The Kid LAROI
        </div>
        """, unsafe_allow_html=True)

    st.balloons()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #00ffff; font-size: 1.1rem;'>
    🔮 <strong>HybridOracle</strong> — Music Vibe Detector 3.0<br>
    <span style='color: #ff00cc;'>Neon Edition • Powered by Perceptron</span>
</div>
""", unsafe_allow_html=True)
