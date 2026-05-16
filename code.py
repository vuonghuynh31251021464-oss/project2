import streamlit as st
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="HybridOracle", layout="wide", initial_sidebar_state="expanded")

# ====================== PASTEL BEAUTIFUL STYLE ======================
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #ffe4f3, #e0f7fa, #f3e0ff, #fff0e6);
        background-attachment: fixed;
        color: #2c2c2c;
    }
    
    .main {
        background: rgba(255, 255, 255, 0.93);
        backdrop-filter: blur(18px);
        border-radius: 30px;
        padding: 3rem;
        border: 2px solid rgba(255, 105, 180, 0.35);
        box-shadow: 0 15px 50px rgba(255, 182, 193, 0.25);
    }
    
    h1 {
        font-size: 3.7rem;
        background: linear-gradient(90deg, #ff69b4, #00ced1, #ba55d3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    h2, h3 { color: #d63384; }
    
    /* Input rõ ràng */
    .stSlider, .stSelectbox {
        background: white;
        padding: 15px;
        border-radius: 20px;
        border: 2px solid #ffb6c1;
        box-shadow: 0 4px 15px rgba(255, 182, 193, 0.2);
    }
    
    /* Button đẹp */
    .stButton>button {
        background: linear-gradient(45deg, #ff69b4, #00ced1, #ba55d3);
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        border-radius: 50px;
        padding: 1rem 3.5rem;
        border: none;
        box-shadow: 0 10px 30px rgba(255, 105, 180, 0.4);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 40px rgba(255, 105, 180, 0.6);
    }
    
    /* Result Box - Nổi bật & Đẹp */
    .result-box {
        background: linear-gradient(135deg, #fff0f5, #e0f7fa);
        padding: 2.8rem;
        border-radius: 28px;
        border: 4px solid #ff69b4;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 45px rgba(255, 105, 180, 0.3);
    }
    
    .vibe-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #d63384;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 HYBRIDORACLE")
st.markdown("### 🎧 **Music Vibe Detector** 🌸")
st.markdown("**AI đoán gu nhạc pastel cho GenZ** ✨")

# Sidebar
with st.sidebar:
    st.markdown("## 🌷 Pastel Vibe AI")
    st.info("Perceptron Model • 24 Samples")
    st.caption("Soft • Sweet • Accurate 💕")

# Utility
def train_perceptron(X, y):
    model = Perceptron(max_iter=2000, tol=1e-3, random_state=42)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)
    return model, scaler

# Input
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
    
    genres = ["Pop", "Rock", "EDM"]
    result = genres[pred]
    
    # Kết quả nổi bật
    st.markdown(f"""
    <div class="result-box">
        <p class="vibe-title">YOUR VIBE IS</p>
        <h1 style="font-size: 4.5rem; margin: 10px 0; color: #d63384;">{result}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    accuracy = model.score(scaler.transform(X), y) * 100
    st.metric("Model Accuracy", f"{accuracy:.1f}%")
    
    # Phân tích chi tiết
    st.markdown("### 🎀 Your Music Personality Analysis")
    
    if pred == 2:      # EDM
        st.success("**🔥 EDM Sweetheart** — Năng lượng bùng nổ!")
        st.write("• Thích bass mạnh, drop cảm xúc và nhịp nhanh")
        st.write("• Thường nghe khi tập gym, party hoặc cần năng lượng")
        st.write("• Tính cách: Năng động, thích thử thách, yêu festival")
        
    elif pred == 1:    # Rock
        st.warning("**🎸 Rock Cutie** — Cá tính mạnh mẽ!")
        st.write("• Thích guitar riff và lời bài hát sâu lắng")
        st.write("• Nghe nhạc để giải tỏa hoặc truyền cảm hứng")
        st.write("• Tính cách: Độc lập, chân thành, đôi khi nổi loạn")
        
    else:              # Pop
        st.info("**✨ Pop Princess / Prince** — Chill & Vui tươi!")
        st.write("• Thích nhạc dễ nghe, bắt tai và theo trend")
        st.write("• Nghe nhạc để thư giãn và cập nhật xu hướng")
        st.write("• Tính cách: Vui vẻ, hòa đồng, dễ thương")

    st.balloons()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #d63384; font-size: 1.1rem;'>
    🔮 <strong>HybridOracle</strong> — Pastel Music Vibe Detector<br>
    Soft Pastel Beautiful Edition • Made for GenZ 💕
</div>
""", unsafe_allow_html=True)
