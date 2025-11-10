import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px # type: ignore

# =========================
# Load model
# 
import sklearn

print("Using scikit-learn version:", sklearn.__version__)
model = joblib.load("best_model.pkl", mmap_mode=None)

# Feature order used during training
FEATURE_ORDER = [
    "distance-to-solar-noon","temperature","wind-direction","wind-speed",
    "sky-cover","visibility","humidity","average-wind-speed-(period)","average-pressure-(period)"
]

# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="Solar Power Prediction", page_icon="☀️", layout="wide")
st.title("⚡ Solar Power Generation Dashboard")
st.write("A machine learning app to predict **solar power generated (Joules)** from environmental data.")

# Sidebar Navigation
menu = st.sidebar.radio("📌 Navigation", ["🔮 Predict", "📂 Batch Upload", "📊 Model Insights"])


# =========================
# PAGE 1: Single Prediction
# =========================
if menu == "🔮 Predict":
    st.header("Single Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        distance_to_solar_noon = st.slider("Distance to Solar Noon (radians)", 0.0, 3.14, 1.5)
        temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
        wind_direction = st.slider("Wind Direction (°)", 0, 360, 180)

    with col2:
        wind_speed = st.number_input("Wind Speed (m/s)", value=3.0)
        sky_cover = st.selectbox("Sky Cover (0 = clear, 4 = fully covered)", [0,1,2,3,4])
        visibility = st.number_input("Visibility (km)", value=10.0)

    with col3:
        humidity = st.slider("Humidity (%)", 0, 100, 50)
        average_wind_speed = st.number_input("Average Wind Speed (m/s)", value=2.5)
        average_pressure = st.number_input("Average Pressure (inHg)", value=29.9)

    if st.button("🚀 Predict"):
        input_data = np.array([[distance_to_solar_noon, temperature, wind_direction, wind_speed,
                                sky_cover, visibility, humidity, average_wind_speed, average_pressure]])
        prediction = model.predict(input_data)[0]
        st.metric("Predicted Power Generated", f"{prediction:,.2f} Joules")


# =========================
# PAGE 2: Batch Upload
# =========================
elif menu == "📂 Batch Upload":
    st.header("Batch Prediction from CSV")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview", df_new.head())

        # Drop target column if present
        if "power_generated" in df_new.columns:
            df_new = df_new.drop("power_generated", axis=1)

        # Validation: Check required columns
        missing_cols = [c for c in FEATURE_ORDER if c not in df_new.columns]
        extra_cols = [c for c in df_new.columns if c not in FEATURE_ORDER]

        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
        else:
            if extra_cols:
                st.warning(f"⚠️ Ignoring extra columns: {extra_cols}")

            # Reorder to match training
            df_new = df_new[FEATURE_ORDER]

            # 🔹 Handle NaN (Imputation)
            # Fill numeric NaN with median
            for col in df_new.columns:
                if df_new[col].isna().sum() > 0:
                    median_val = df_new[col].median()
                    df_new[col] = df_new[col].fillna(median_val)

            # Predictions
            preds = model.predict(df_new)
            df_new["Predicted_Power"] = preds

            st.success("✅ Predictions Completed!")
            st.write(df_new.head())

            # Download
            csv = df_new.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv,
                               file_name="predictions.csv", mime="text/csv")



# =========================
# PAGE 3: Model Insights
# =========================
elif menu == "📊 Model Insights":
    st.header("Model Insights")

    # Example metrics (replace with saved metrics if available)
    st.metric("R² Score", "0.92")
    st.metric("RMSE", "3034")

    # Feature Importance (for tree-based models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": FEATURE_ORDER, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=True)

        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     title="Feature Importance", color="Importance",
                     color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Feature importance not available for this model.")
