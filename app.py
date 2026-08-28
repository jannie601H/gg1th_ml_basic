import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ==================================================
# Page Config
# ==================================================
st.set_page_config(
    page_title="EV Speed Advisor",
    page_icon="⚡",
    layout="wide"
)


# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "datas_ml" / "ev_energy_consumption.csv"
MODEL_PATH = BASE_DIR / "xgb_model.pkl"


# ==================================================
# Load Data / Model
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
xgb_model = load_model()


# ==================================================
# Global CSS
# ==================================================
st.html("""
<style>

.block-container {
    max-width: 1200px;
    padding-top: 4rem;
    padding-bottom: 3rem;
}

.ev-title {
    font-size: 2.5rem;
    font-weight: 750;
    letter-spacing: -0.03em;
    margin-bottom: 0.3rem;
}

.ev-subtitle {
    color: #888;
    font-size: 1rem;
    margin-bottom: 2.2rem;
}

/* Result card */
.result-card {
    border: 1px solid rgba(128, 128, 128, 0.3);
    border-radius: 18px;
    padding: 28px 20px;
    text-align: center;
    margin-top: 10px;
    min-height: 245px;
}

.mode-name {
    color: #888;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin-bottom: 18px;
}

.speed-value {
    font-size: 2.8rem;
    font-weight: 750;
    line-height: 1.1;
    margin-bottom: 20px;
}

.speed-unit {
    color: #888;
    font-size: 1rem;
    font-weight: 500;
}

.result-detail {
    color: #777;
    font-size: 0.9rem;
    line-height: 1.65;
}

.result-detail strong {
    color: inherit;
    font-size: 1rem;
}

.tolerance-label {
    display: inline-block;
    margin-top: 14px;
    padding: 5px 11px;
    border: 1px solid rgba(128, 128, 128, 0.3);
    border-radius: 100px;
    font-size: 0.78rem;
}

/* Main button */
div.stButton > button {
    width: 100%;
    height: 3.2rem;
    border-radius: 12px;
    font-weight: 650;
}

</style>
""")


# ==================================================
# Prediction
# ==================================================
def predict_speed_profile(
    model,
    distance,
    road_grade,
    payload,
    ambient_temp,
    hvac_power,
    battery_temp,
    driving_style,
    tire_pressure
):

    speeds = np.arange(20, 131, 1)

    results = []

    for speed in speeds:

        X_input = pd.DataFrame([{
            "speed_kmh": speed,
            "payload_kg": payload,
            "ambient_temp_C": ambient_temp,
            "hvac_power_kw": hvac_power,
            "road_grade_pct": road_grade,
            "battery_temp_C": battery_temp,
            "driving_style_index": driving_style,
            "tire_pressure_bar": tire_pressure,
            "trip_distance_km": distance
        }])

        # kWh / 100 km
        consumption = model.predict(X_input)[0]

        # 실제 주행 거리에서 사용하는 총 에너지
        total_energy = consumption * distance / 100

        # hour
        travel_time = distance / speed

        results.append([
            speed,
            consumption,
            total_energy,
            travel_time
        ])

    return pd.DataFrame(
        results,
        columns=[
            "speed",
            "consumption_per_100km",
            "total_energy",
            "travel_time"
        ]
    )


# ==================================================
# Mode Selection
# ==================================================
def select_mode_speed(result, energy_tolerance):

    min_energy = result["total_energy"].min()

    energy_limit = (
        min_energy
        * (1 + energy_tolerance)
    )

    allowed = result[
        result["total_energy"]
        <= energy_limit
    ]

    # 허용 범위 내에서 가장 빠른 속도
    idx = allowed["speed"].idxmax()

    return allowed.loc[idx]


# ==================================================
# Result Card
# ==================================================
def show_mode_card(
    name,
    row,
    tolerance
):

    html = f"""
<div class="result-card">
    <div class="mode-name">{name}</div>

    <div class="speed-value">
        {row["speed"]:.0f}
        <span class="speed-unit">km/h</span>
    </div>

    <div class="result-detail">

        Energy Consumption<br>
        <strong>
            {row["consumption_per_100km"]:.2f} kWh/100km
        </strong>

        <br><br>

        Total Energy<br>
        <strong>
            {row["total_energy"]:.2f} kWh
        </strong>

        <br><br>

        Travel Time<br>
        <strong>
            {row["travel_time"] * 60:.0f} min
        </strong>

        <br>

        <span class="tolerance-label">
            Energy tolerance +{tolerance}%
        </span>

    </div>
</div>
"""

    # 핵심: st.markdown이 아니라 st.html 사용
    st.html(html)


# ==================================================
# Header
# ==================================================
st.html("""
<div class="ev-title">
    EV Speed Advisor
</div>

<div class="ev-subtitle">
    주행 조건에 따른 전기차의 에너지 소비량을 예측하고
    효율적인 주행 속도를 추천한다.
</div>
""")


# ==================================================
# Input
# ==================================================
st.subheader("Driving Conditions")

left, right = st.columns(
    2,
    gap="large"
)


with left:

    distance = st.slider(
        "Trip Distance",
        min_value=5,
        max_value=200,
        value=100,
        step=5,
        format="%d km"
    )

    payload = st.slider(
        "Payload",
        min_value=0,
        max_value=500,
        value=150,
        step=10,
        format="%d kg"
    )


with right:

    road_grade = st.slider(
        "Road Grade",
        min_value=-5.0,
        max_value=8.0,
        value=0.0,
        step=0.1,
        format="%.1f %%"
    )

    hvac_power = st.slider(
        "HVAC Power",
        min_value=0.0,
        max_value=5.0,
        value=1.5,
        step=0.1,
        format="%.1f kW"
    )


# ==================================================
# Default Values
# ==================================================
ambient_temp_default = float(
    df["ambient_temp_C"].median()
)

battery_temp_default = float(
    df["battery_temp_C"].median()
)

driving_style_default = float(
    df["driving_style_index"].median()
)

tire_pressure_default = float(
    df["tire_pressure_bar"].median()
)


# ==================================================
# Advanced Settings
# ==================================================
with st.expander("Advanced Settings"):

    st.caption(
        "기본값은 학습 데이터의 중앙값이다."
    )

    adv_left, adv_right = st.columns(2)

    with adv_left:

        ambient_temp = st.slider(
            "Ambient Temperature",
            min_value=float(
                df["ambient_temp_C"].min()
            ),
            max_value=float(
                df["ambient_temp_C"].max()
            ),
            value=ambient_temp_default,
            step=0.5,
            format="%.1f °C"
        )

        battery_temp = st.slider(
            "Battery Temperature",
            min_value=float(
                df["battery_temp_C"].min()
            ),
            max_value=float(
                df["battery_temp_C"].max()
            ),
            value=battery_temp_default,
            step=0.5,
            format="%.1f °C"
        )


    with adv_right:

        driving_style = st.slider(
            "Driving Style Index",
            min_value=0.0,
            max_value=1.0,
            value=driving_style_default,
            step=0.05
        )

        tire_pressure = st.slider(
            "Tire Pressure",
            min_value=float(
                df["tire_pressure_bar"].min()
            ),
            max_value=float(
                df["tire_pressure_bar"].max()
            ),
            value=tire_pressure_default,
            step=0.05,
            format="%.2f bar"
        )


# ==================================================
# Button
# ==================================================
st.write("")

recommend = st.button(
    "Find Recommended Speed",
    type="primary"
)


# ==================================================
# Recommendation
# ==================================================
if recommend:

    result = predict_speed_profile(
        model=xgb_model,
        distance=distance,
        road_grade=road_grade,
        payload=payload,
        ambient_temp=ambient_temp,
        hvac_power=hvac_power,
        battery_temp=battery_temp,
        driving_style=driving_style,
        tire_pressure=tire_pressure
    )


    # -------------------------------
    # Mode calculation
    # -------------------------------
    perfect = select_mode_speed(
        result,
        0.00
    )

    eco = select_mode_speed(
        result,
        0.10
    )

    balanced = select_mode_speed(
        result,
        0.15
    )

    fast = select_mode_speed(
        result,
        0.25
    )


    # ==================================================
    # Result Cards
    # ==================================================
    st.divider()

    st.subheader(
        "Recommended Speed"
    )

    st.caption(
        "최소 에너지 소비량 대비 허용 가능한 "
        "추가 에너지 소비량에 따라 속도를 추천한다."
    )


    col1, col2, col3 = st.columns(
        3,
        gap="medium"
    )


    with col1:

        show_mode_card(
            "ECO",
            eco,
            10
        )


    with col2:

        show_mode_card(
            "BALANCED",
            balanced,
            15
        )


    with col3:

        show_mode_card(
            "FAST",
            fast,
            25
        )


    # ==================================================
    # Speed / Energy Chart
    # ==================================================
    st.write("")
    st.write("")

    st.subheader(
        "Speed & Energy Profile"
    )

    st.caption(
        "현재 주행 조건에서 속도 변화에 따른 "
        "예상 총 에너지 소비량이다."
    )


    chart_data = (
        result[
            [
                "speed",
                "total_energy"
            ]
        ]
        .set_index("speed")
    )


    st.line_chart(
        chart_data,
        x_label="Speed (km/h)",
        y_label="Total Energy (kWh)"
    )


    # ==================================================
    # Mode Comparison
    # ==================================================
    st.write("")

    st.subheader(
        "Mode Comparison"
    )


    comparison = pd.DataFrame({

        "Mode": [
            "Energy Optimum",
            "Eco",
            "Balanced",
            "Fast"
        ],

        "Speed (km/h)": [
            perfect["speed"],
            eco["speed"],
            balanced["speed"],
            fast["speed"]
        ],

        "Energy (kWh)": [
            perfect["total_energy"],
            eco["total_energy"],
            balanced["total_energy"],
            fast["total_energy"]
        ],

        "Travel Time (min)": [
            perfect["travel_time"] * 60,
            eco["travel_time"] * 60,
            balanced["travel_time"] * 60,
            fast["travel_time"] * 60
        ]
    })


    comparison["Speed (km/h)"] = (
        comparison["Speed (km/h)"]
        .round()
        .astype(int)
    )

    comparison["Energy (kWh)"] = (
        comparison["Energy (kWh)"]
        .round(2)
    )

    comparison["Travel Time (min)"] = (
        comparison["Travel Time (min)"]
        .round(1)
    )


    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True
    )


    # ==================================================
    # Absolute Energy Optimum
    # ==================================================
    with st.expander(
        "Absolute Energy Optimum"
    ):

        st.metric(
            "Recommended Speed",
            f"{perfect['speed']:.0f} km/h"
        )

        col_a, col_b, col_c = st.columns(3)

        with col_a:

            st.metric(
                "Consumption",
                (
                    f"{perfect['consumption_per_100km']:.2f} "
                    "kWh/100km"
                )
            )

        with col_b:

            st.metric(
                "Total Energy",
                f"{perfect['total_energy']:.2f} kWh"
            )

        with col_c:

            st.metric(
                "Travel Time",
                f"{perfect['travel_time'] * 60:.1f} min"
            )