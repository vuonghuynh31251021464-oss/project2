import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="HybridOracle", layout="wide")

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
    font-size: 1rem;
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🔮 HybridOracle")
st.markdown("### 🧩 Hybrid Intelligent Prediction System  \n"
            "*Kết hợp Perceptron Neural Network & Fuzzy Logic*")

menu = st.sidebar.radio("Select Prediction Task:",
                        ["Air Pollution", "Soil Classification", "Music Preference",
                         "Stress Level", "Traffic Forecast"])

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
# MODEL 1 - Air Pollution
# ========================
if menu == "Air Pollution":
    st.header("🌫️ Air Pollution Classification")
    pm25 = st.slider("PM2.5 Fine Dust (µg/m³)", 0, 300, 50)
    co2 = st.slider("CO₂ Concentration (ppm)", 300, 2000, 600)
    humidity = st.slider("Humidity (%)", 0, 100, 40)
    
    # Fuzzy Variables
    pm25_in = ctrl.Antecedent(np.arange(0, 301, 1), 'pm25')
    co2_in = ctrl.Antecedent(np.arange(300, 2001, 1), 'co2')
    hum_in = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    quality_out = ctrl.Consequent(np.arange(0, 3, 1), 'quality')

    # Membership functions
    pm25_in['low'] = fuzz.trapmf(pm25_in.universe, [0,0,50,100])
    pm25_in['med'] = fuzz.trapmf(pm25_in.universe, [80,100,150,200])
    pm25_in['high'] = fuzz.trapmf(pm25_in.universe, [180,250,300,300])
    
    co2_in['low'] = fuzz.trimf(co2_in.universe, [300,400,800])
    co2_in['high'] = fuzz.trimf(co2_in.universe, [700,1500,2000])
    
    hum_in['low'] = fuzz.trimf(hum_in.universe, [0,30,50])
    hum_in['med'] = fuzz.trimf(hum_in.universe, [40,60,80])
    hum_in['high'] = fuzz.trimf(hum_in.universe, [70,90,100])

    quality_out['good'] = fuzz.trimf(quality_out.universe, [0,0,1])
    quality_out['average'] = fuzz.trimf(quality_out.universe, [0,1,2])
    quality_out['hazard'] = fuzz.trimf(quality_out.universe, [1,2,2])

    # Rules
    rules = [
        ctrl.Rule(pm25_in['low'] & co2_in['low'], quality_out['good']),
        ctrl.Rule(pm25_in['med'] | hum_in['med'], quality_out['average']),
        ctrl.Rule(pm25_in['high'] | co2_in['high'], quality_out['hazard']),
    ]

    sys = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(sys)
    sim.input['pm25'] = pm25
    sim.input['co2'] = co2
    sim.input['humidity'] = humidity
    sim.compute()
    
    score = sim.output['quality']
    result = ["Good", "Average", "Hazardous"][int(round(score))]
    
    st.success(f"**Predicted Air Quality:** {result}")
    st.metric("Fuzzy Output Score", round(score,2))

# ========================
# MODEL 2 - Soil Classification
# ========================
elif menu == "Soil Classification":
    st.header("🌱 Soil Classification")
    # (Giữ nguyên code phần này của bạn)
    ph = st.slider("Soil pH", 3.0, 9.0, 6.5)
    nitrogen = st.slider("Nitrogen content (%)", 0.0, 1.0, 0.3)
    humidity = st.slider("Humidity (%)", 0, 100, 40)

    X = np.array([
        [5.5, 0.2, 60],
        [6.5, 0.5, 50],
        [7.5, 0.3, 40],
        [6.0, 0.1, 70],
        [5.8, 0.6, 80]
    ])
    y = np.array([0, 2, 1, 0, 2])
    
    model, scaler = train_perceptron(X, y)
    pred = model.predict(scaler.transform([[ph, nitrogen, humidity]]))[0]
    result = ["Good Soil for Rice", "Good Soil for Corn", "Good Soil for Vegetables"][pred]
    
    st.success(f"**Predicted Result:** {result}")

# ========================
# MODEL 3 - Music Preference
# ========================
elif menu == "Music Preference":
    st.header("🎧 Identify Your Favorite Music")
    # (Giữ nguyên code phần này)
    age = st.slider("Age", 10, 70, 25)
    hours = st.slider("Hours listening to music/day", 0, 10, 3)
    habit = st.selectbox("Listening Habits", ["Radio", "Spotify", "YouTube"])
    habit_code = {"Radio":0, "Spotify":1, "YouTube":2}[habit]

    X = np.array([
        [20,4,1],[30,2,0],[25,6,2],
        [40,1,0],[18,8,2],[35,3,1]
    ])
    y = np.array([2,1,2,1,2,0])
    
    model, scaler = train_perceptron(X, y)
    pred = model.predict(scaler.transform([[age,hours,habit_code]]))[0]
    result = ["Pop","Rock","EDM"][pred]
    
    st.success(f"🎵 Your predicted favorite music: **{result}**")

# ========================
# MODEL 4 - Stress Level
# ========================
elif menu == "Stress Level":
    st.header("😰 Stress Level Classification")
    # (Giữ nguyên phần code Stress Level của bạn)
    work_hours = st.slider("Hours of work/day", 0, 16, 8)
    sleep_hours = st.slider("Hours of sleep/day", 0, 12, 6)
    coffee = st.slider("Cups of coffee/day", 0, 10, 2)
    
    # Fuzzy code phần Stress Level...
    stress_in = ctrl.Antecedent(np.arange(0, 17, 1), 'work')
    sleep_in = ctrl.Antecedent(np.arange(0, 13, 1), 'sleep')
    coffee_in = ctrl.Antecedent(np.arange(0, 11, 1), 'coffee')
    stress_out = ctrl.Consequent(np.arange(0, 2, 1), 'stress')

    stress_in['low'] = fuzz.trapmf(stress_in.universe, [0,0,6,8])
    stress_in['high'] = fuzz.trapmf(stress_in.universe, [8,10,16,16])
    sleep_in['low'] = fuzz.trapmf(sleep_in.universe, [0,0,5,6])
    sleep_in['good'] = fuzz.trapmf(sleep_in.universe, [6,7,9,10])
    coffee_in['few'] = fuzz.trapmf(coffee_in.universe, [0,0,2,3])
    coffee_in['many'] = fuzz.trapmf(coffee_in.universe, [3,5,10,10])

    stress_out['normal'] = fuzz.trimf(stress_out.universe, [0,0,1])
    stress_out['malnutrition'] = fuzz.trimf(stress_out.universe, [0,1,1])

    rules = [
        ctrl.Rule(stress_in['high'] & sleep_in['low'] & coffee_in['many'], stress_out['malnutrition']),
        ctrl.Rule(stress_in['low'] | sleep_in['good'], stress_out['normal']),
    ]

    sys = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(sys)
    sim.input['work'] = work_hours
    sim.input['sleep'] = sleep_hours
    sim.input['coffee'] = coffee
    sim.compute()
    
    score = sim.output['stress']
    result = "Normal" if score < 0.5 else "Malnutrition/Stress"
    
    st.success(f"☯️ Stress Level: **{result}**")
    st.metric("Fuzzy Score", round(score,2))

# ========================
# MODEL 5 - Traffic Forecast
# ========================
elif menu == "Traffic Forecast":
    st.header("🚗 Traffic Forecast")
    # (Giữ nguyên code phần Traffic của bạn)
    hour = st.slider("Hour of Day (0–23)", 0, 23, 8)
    vehicles = st.slider("Vehicles per minute", 0, 100, 30)
    weather = st.selectbox("Weather", ["Sunny","Rainy"])
    weather_code = {"Sunny":0, "Rainy":1}[weather]

    # Fuzzy code...
    traffic_in = ctrl.Antecedent(np.arange(0, 101, 1), 'vehicles')
    hour_in = ctrl.Antecedent(np.arange(0, 24, 1), 'hour')
    traffic_out = ctrl.Consequent(np.arange(0,3,1), 'traffic')

    traffic_in['low'] = fuzz.trapmf(traffic_in.universe, [0,0,30,40])
    traffic_in['med'] = fuzz.trapmf(traffic_in.universe, [30,40,60,70])
    traffic_in['high'] = fuzz.trapmf(traffic_in.universe, [60,80,100,100])

    hour_in['morning'] = fuzz.trimf(hour_in.universe, [6,8,10])
    hour_in['noon'] = fuzz.trimf(hour_in.universe, [11,13,15])
    hour_in['evening'] = fuzz.trimf(hour_in.universe, [16,18,20])
    hour_in['night'] = fuzz.trimf(hour_in.universe, [21,23,23])

    traffic_out['clear'] = fuzz.trimf(traffic_out.universe, [0,0,1])
    traffic_out['medium'] = fuzz.trimf(traffic_out.universe, [0,1,2])
    traffic_out['congested'] = fuzz.trimf(traffic_out.universe, [1,2,2])

    rules = [
        ctrl.Rule(traffic_in['low'], traffic_out['clear']),
        ctrl.Rule(hour_in['morning'] | hour_in['evening'] | traffic_in['high'], traffic_out['congested']),
        ctrl.Rule(traffic_in['med'], traffic_out['medium'])
    ]

    sys = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(sys)
    sim.input['vehicles'] = vehicles
    sim.input['hour'] = hour
    sim.compute()
    
    score = sim.output['traffic']
    result = ["Clear", "Medium", "Congested"][int(round(score))]
    
    st.success(f"🚦 Predicted Traffic: **{result}**")
    st.metric("Fuzzy output", round(score,2))

st.caption("🔮 HybridOracle • Developed with ❤️ using Streamlit, Scikit-learn, and Fuzzy Logic")
