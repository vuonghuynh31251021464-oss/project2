import streamlit as st
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="HybridOracle - Music Preference", layout="wide")

st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}
.main {
    background-color: white;
    padding: 2rem;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #1E3A8A;
}
.stButton>button {
    color: white;
    background-color: #1E3A8A;
    border-radius: 8px;
    border: none;
    font-size: 1.1rem;
    padding: 0.6rem 1.2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🔮 HybridOracle")
st.markdown("### 🧩 Hybrid Intelligent Prediction System")
st.markdown("### 🎧 Music Preference Prediction")

# Sidebar
menu = st.sidebar.radio("Select Prediction Task:", ["Identify Your Favorite Music"])

# ----------------------------
# Utility Function for Perceptron
# ----------------------------
def train_perceptron(X, y):
    model = Perceptron(max_iter=1000, tol=1e-3)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)
    return model, scaler

# ========================
# MUSIC PREFERENCE MODEL (Giữ nguyên code gốc của bạn)
# ========================
if menu == "Identify Your Favorite Music":
    st.header("🎧 Identify Your Favorite Music")
    
    age = st.slider("Age", 10, 70, 25)
    hours = st.slider("Hours listening to music/day", 0, 10, 3)
    habit = st.selectbox("Listening Habits", ["Radio", "Spotify", "YouTube"])
    habit_code = {"Radio":0, "Spotify":1, "YouTube":2}[habit]

    # Fake small dataset (giữ nguyên)
    X = np.array([
        [20,4,1],[30,2,0],[25,6,2],
        [40,1,0],[18,8,2],[35,3,1]
    ])
    y = np.array([2,1,2,1,2,0]) # 0=Pop,1=Rock,2=EDM

    # Nút Predict theo yêu cầu assignment
    if st.button("🔍 Predict My Favorite Music", type="primary", use_container_width=True):
        model, scaler = train_perceptron(X, y)
        pred = model.predict(scaler.transform([[age, hours, habit_code]]))[0]
        
        result = ["Pop", "Rock", "EDM"][pred]
        
        st.success(f"🎵 **Your predicted favorite music: {result}**")
        
        # Thêm thông tin chi tiết
        st.info(f"Age: {age} | Listening: {hours} hours/day | Habit: {habit}")

    st.caption("Model trained with Perceptron on sample dataset")

st.caption("🔮 HybridOracle • Music Preference Prediction using Perceptron")
