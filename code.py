import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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
</style>
""", unsafe_allow_html=True)

st.title("🎧 AlgoRhythm 🎵")
st.markdown("### 🎧 **Music Vibe Detector 3.0** ✨")
st.markdown("**AI đoán gu nhạc GenZ chỉ trong tích tắc** 🔥")

# Sidebar
with st.sidebar:
    st.markdown("## ⚡ VIBE CHECK ACTIVE")
    st.markdown("**Music Taste AI v3.0**")
    st.info("• Perceptron Model\n• CSV Dataset\n• GenZ Neon Edition")
    st.markdown("---")
    st.caption("Made with neon & love for GenZ")

# ================= LOAD DATA =================
data = pd.read_csv("algorhythm_music_vibe_data.csv")

# ===== RENAME COLUMNS =====
data.columns = data.columns.str.strip()

data = data.rename(columns={
    "Age": "age",
    "Hours_Per_Day": "hours",
    "Main_Platform": "platform",
    "Genre": "genre"
})

# ===== MAP TEXT -> NUMBER =====
platform_map = {
    "Radio": 0,
    "Spotify": 1,
    "YouTube": 2,
    "TikTok": 3
}

genre_map = {
    "Pop": 0,
    "Rock": 1,
    "EDM": 2
}

data["platform"] = data["platform"].map(platform_map)
data["genre"] = data["genre"].map(genre_map)

# ===== CLEAN DATA =====
data["age"] = pd.to_numeric(data["age"], errors="coerce")
data["hours"] = pd.to_numeric(data["hours"], errors="coerce")

data = data.dropna()

# DEBUG
st.write("✅ Cleaned Data:", data.head())
# ================= MODEL =================
X = data[["age","hours","platform"]].values
y = data["genre"].values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

model = Perceptron(max_iter=2000, tol=1e-3, random_state=42)
model.fit(Xs, y)

# ================= INPUT =================
st.header("🎧 What's Your Music Vibe Today?")

c1, c2, c3 = st.columns([1,1,1])

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
    habit_display = st.selectbox("📱 Main Platform", list(platform_options.keys()))
    habit_code = platform_options[habit_display]

# ================= PREDICT =================
if st.button("🔥 GUESS MY VIBE NOW", type="primary", use_container_width=True):

    # ===== PREDICT =====
    input_data = np.array([[age, hours, habit_code]])
    pred = model.predict(scaler.transform(input_data))[0]

    genres = ["🎤 Pop", "🎸 Rock", "🔥 EDM"]
    result = genres[pred]

    st.markdown(f"""
    <div class="result-box">
        <h2>YOUR VIBE IS</h2>
        <h1 style="font-size: 4.5rem;">{result}</h1>
    </div>
    """, unsafe_allow_html=True)

    accuracy = model.score(Xs, y) * 100
    st.metric("Model Accuracy", f"{accuracy:.1f}%")

    # ===== ANALYSIS CARD =====
    st.markdown("### 🎯 Your Music Personality Analysis")

    if pred == 2:
        st.success("**🔥 EDM OVERLOAD** — Bạn là **Party Animal**!")
    elif pred == 1:
        st.warning("**🎸 ROCK REBEL** — Bạn là **Emotional Rocker**!")
    else:
        st.info("**✨ POP VIBES** — Bạn là **Trendy Pop Lover**!")

    st.balloons()

    # ====================== DATA ANALYSIS ======================
    st.markdown("---")
    st.markdown("## 📊 Phân tích dữ liệu khảo sát thực tế")

    st.subheader("👀 Dataset Preview")
    st.dataframe(data.head())

    st.subheader("📈 Statistics")
    st.write(data.describe())

    # Histogram
    st.subheader("📊 Distribution")

    fig1 = data.hist(figsize=(8,6))
    st.pyplot(plt.gcf())

    # Heatmap
    st.subheader("🔥 Correlation Heatmap")
    fig2 = plt.figure()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig2)

    # ===== MODEL EVALUATION =====
    st.subheader("🤖 Model Evaluation")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler2 = StandardScaler()
    X_train = scaler2.fit_transform(X_train)
    X_test = scaler2.transform(X_test)

    model2 = Perceptron(max_iter=2000)
    model2.fit(X_train, y_train)

    train_acc = model2.score(X_train, y_train)*100
    test_acc = model2.score(X_test, y_test)*100

    c1, c2 = st.columns(2)
    c1.metric("Train Accuracy", f"{train_acc:.2f}%")
    c2.metric("Test Accuracy", f"{test_acc:.2f}%")

    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")

    cm = confusion_matrix(y_test, model2.predict(X_test))

    fig3 = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig3)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #00ffff;'>
    🎧 AlgoRhythm 🎵 — Music Vibe Detector 3.0
    </div>
    """, unsafe_allow_html=True)

