import streamlit as st
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="HybridOracle - Music Preference", layout="wide")

st.markdown("""
<style>
    body { background-color: #f7f9fc; }
    .main { background-color: white; padding: 2rem; border-radius: 10px; }
    h1, h2, h3 { color: #1E3A8A; }
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
st.markdown("### 🎧 Music Preference Prediction System")
st.markdown("**Perceptron Machine Learning Model**")

# ====================== SIDEBAR ======================
st.sidebar.header("ℹ️ Model Information")
st.sidebar.markdown("""
**Model:** Perceptron  
**Algorithm:** Single Layer Neural Network  
**Library:** scikit-learn  
**Dataset:** 24 mẫu dữ liệu giả lập  
**Features:** Age, Listening Hours, Listening Habit  
**Target:** Music Genre (Pop, Rock, EDM)
""")
st.sidebar.info("Mô hình học từ dữ liệu để dự đoán thể loại nhạc yêu thích của bạn.")

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
# MAIN MUSIC PREFERENCE MODEL
# ========================
st.header("🎧 Identify Your Favorite Music")

st.markdown("""
Hệ thống sẽ dự đoán **thể loại nhạc yêu thích** của bạn dựa trên:
- Tuổi tác  
- Thời gian nghe nhạc mỗi ngày  
- Thói quen nghe nhạc
""")

# Input Fields
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Your Age", 10, 70, 25)
    hours = st.slider("Hours listening to music per day", 0, 12, 3)

with col2:
    habit = st.selectbox("Main Listening Habit", ["Radio", "Spotify", "YouTube"])
    habit_code = {"Radio": 0, "Spotify": 1, "YouTube": 2}[habit]

# ==================== 24 MẪU DỮ LIỆU ====================
X = np.array([
    [20,4,1], [30,2,0], [25,6,2], [40,1,0], [18,8,2], [35,3,1],
    [22,5,2], [28,3,1], [33,7,2], [45,2,0], [19,9,2], [38,4,1],
    [24,6,2], [29,2,0], [26,5,2], [42,1,1], [21,7,2], [31,3,0],
    [27,8,2], [36,2,1], [23,6,2], [39,4,0], [20,5,2], [34,3,1]
])

y = np.array([2,1,2,1,2,0, 2,1,2,1,2,0, 2,1,2,1,2,0, 2,1,2,1,2,0])

# Predict Button
if st.button("🔍 Predict My Favorite Music", type="primary", use_container_width=True):
    
    model, scaler = train_perceptron(X, y)
    
    input_data = np.array([[age, hours, habit_code]])
    pred = model.predict(scaler.transform(input_data))[0]
    
    genres = ["Pop", "Rock", "EDM"]
    result = genres[pred]
    
    # Kết quả chính
    st.success(f"🎵 **Your predicted favorite music genre: {result}**")
    
    # Độ chính xác
    accuracy = model.score(scaler.transform(X), y) * 100
    st.metric(label="Model Training Accuracy", value=f"{accuracy:.2f}%")
    
    # Phân tích cá nhân hóa
    st.markdown("### 📊 Phân tích cá nhân hóa")
    st.write(f"**• Tuổi:** {age} tuổi")
    st.write(f"**• Thời gian nghe:** {hours} giờ/ngày")
    st.write(f"**• Thói quen:** {habit}")
    
    if result == "EDM":
        st.info("🔥 Bạn có xu hướng thích nhạc điện tử sôi động, nhịp nhanh. Thường là người năng động, yêu thích party và festival.")
    elif result == "Rock":
        st.info("🎸 Bạn nghiêng về Rock - thể loại mạnh mẽ, giàu cảm xúc và cá tính. Thường là người sâu lắng và đam mê.")
    else:
        st.info("🎤 Bạn thích Pop - thể loại dễ nghe, bắt tai, vui tươi và phổ biến rộng rãi.")
    
    st.caption("Lưu ý: Đây là mô hình minh họa sử dụng dữ liệu giả lập để demo Perceptron.")

st.markdown("---")
st.caption("🔮 HybridOracle • Music Preference Prediction using Perceptron")
st.caption("Dataset: 24 samples | Algorithm: Perceptron | Accuracy calculated on training data")
