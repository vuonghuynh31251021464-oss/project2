import streamlit as st
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="HybridOracle - Music Preference", layout="wide")

st.markdown("""
<style>
body { background-color: #f7f9fc; }
.main { background-color: white; padding: 2rem; border-radius: 10px; }
h1, h2, h3 { color: #1E3A8A; }
.stButton>button {
    color: white; background-color: #1E3A8A;
    border-radius: 8px; border: none; font-size: 1.1rem; padding: 0.6rem 1.2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🔮 HybridOracle")
st.markdown("### 🎧 Music Preference Prediction System")
st.markdown("**Perceptron Model - Machine Learning**")

# Sidebar
st.sidebar.header("About this Model")
st.sidebar.info("Mô hình sử dụng Perceptron để dự đoán thể loại nhạc yêu thích dựa trên tuổi, thời gian nghe nhạc và thói quen nghe.")

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
st.header("🎧 Identify Your Favorite Music")

st.markdown("""
**Hệ thống sẽ dự đoán thể loại nhạc bạn yêu thích nhất**  
Dựa trên: Tuổi • Thời gian nghe nhạc mỗi ngày • Thói quen nghe nhạc
""")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Your Age", 10, 70, 25)
    hours = st.slider("Hours listening to music per day", 0, 12, 3)

with col2:
    habit = st.selectbox("Main Listening Habit", ["Radio", "Spotify", "YouTube"])
    habit_code = {"Radio": 0, "Spotify": 1, "YouTube": 2}[habit]

# ==================== Expanded Dataset ====================
X = np.array([
    [20,4,1], [30,2,0], [25,6,2], [40,1,0], [18,8,2], [35,3,1],
    [22,5,2], [28,3,1], [33,7,2], [45,2,0], [19,9,2], [38,4,1],
    [24,6,2], [29,2,0], [26,5,2], [42,1,1], [21,7,2], [31,3,0],
    [27,8,2], [36,2,1], [23,6,2], [39,4,0], [20,5,2], [34,3,1]
])

y = np.array([2,1,2,1,2,0, 2,1,2,1,2,0, 2,1,2,1,2,0, 2,1,2,1,2,0])

# Button Predict
if st.button("🔍 Predict My Favorite Music", type="primary", use_container_width=True):
    
    model, scaler = train_perceptron(X, y)
    
    input_data = np.array([[age, hours, habit_code]])
    pred = model.predict(scaler.transform(input_data))[0]
    
    genres = ["Pop", "Rock", "EDM"]
    result = genres[pred]
    
    # Hiển thị kết quả chính
    st.success(f"🎵 **Your predicted favorite music genre: {result}**")
    
    # Hiển thị độ chính xác
    accuracy = model.score(scaler.transform(X), y) * 100
    st.metric("Model Training Accuracy", f"{accuracy:.2f}%")
    
    # Giải thích thêm
    st.markdown("### 📋 Phân tích dự đoán:")
    st.write(f"• **Age**: {age} tuổi")
    st.write(f"• **Listening Time**: {hours} giờ/ngày")
    st.write(f"• **Habit**: {habit}")
    
    if result == "EDM":
        st.info("🔥 Bạn có xu hướng thích nhạc sôi động, nhịp nhanh. Thường là người năng động!")
    elif result == "Rock":
        st.info("🎸 Bạn nghiêng về thể loại Rock - mạnh mẽ, cảm xúc và sâu lắng.")
    else:
        st.info("🎤 Bạn thích Pop - dễ nghe, bắt tai và phổ biến.")
    
    st.caption("Lưu ý: Đây là mô hình minh họa sử dụng dữ liệu giả lập.")

st.markdown("---")
st.caption("🔮 HybridOracle • Music Preference Prediction using Perceptron")
st.caption("Dataset: 24 samples | Algorithm: Perceptron | Library: scikit-learn")
