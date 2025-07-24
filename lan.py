import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load model components
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# UI Configuration
st.set_page_config(page_title="Rain Predictor", page_icon="🌧️")
st.title("🌦️ Smart Rain Prediction")
st.markdown("Predict tomorrow's weather using key meteorological parameters")
# Form Elements
with st.form("weather_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        date = st.date_input("Date", value=datetime.today())
        location = st.selectbox("Location", label_encoders['Location'].classes_)
        mintemp = st.number_input("Min Temp (°C)", min_value=-10, max_value=50, value=14)
        maxtemp = st.number_input("Max Temp (°C)", min_value=-10, max_value=50, value=22)
        humidity_9am = st.slider("9 AM Humidity (%)", 0, 100, 70)
        pressure_9am = st.number_input("9 AM Pressure (hPa)", value=1013)

    with col2:
        wind_gust_dir = st.selectbox("Wind Gust Direction", label_encoders['WindGustDir'].classes_)
        wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", value=33)
        wind_speed_9am = st.number_input("9 AM Wind Speed (km/h)", value=11)
        wind_speed_3pm = st.number_input("3 PM Wind Speed (km/h)", value=19)
        humidity_3pm = st.slider("3 PM Humidity (%)", 0, 100, 50)
        pressure_3pm = st.number_input("3 PM Pressure (hPa)", value=1010)
        rain_today = st.radio("Rain Today?", label_encoders['RainToday'].classes_)

    submitted = st.form_submit_button("Predict Weather")

if submitted:
    try:
        # Create input dictionary
        input_data = {
            "Location": location,
            "MinTemp": mintemp,
            "MaxTemp": maxtemp,
            "Humidity9am": humidity_9am,
            "Humidity3pm": humidity_3pm,
            "Pressure9am": pressure_9am,
            "Pressure3pm": pressure_3pm,
            "WindGustDir": wind_gust_dir,
            "WindGustSpeed": wind_gust_speed,
            "WindSpeed9am": wind_speed_9am,
            "WindSpeed3pm": wind_speed_3pm,
            "RainToday": rain_today,
            "Year": date.year,
            "Month": date.month,
            "Day": date.day
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Label encoding
        categorical_cols = ['Location', 'WindGustDir', 'RainToday']
        for col in categorical_cols:
            le = label_encoders[col]
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])

        # Ensure all model-expected features
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Add missing features with default

        input_df = input_df[feature_names]  # Maintain feature order

        # Scale and predict
        scaled_input = scaler.transform(input_df)
        prob_rain = model.predict_proba(scaled_input)[0][1]
        
        # Display results
        st.subheader("Prediction Result")
        if prob_rain > 0.5 and prob_rain <0.75:
            st.success(f"chance to get rain({prob_rain:.1%})")
        elif(prob_rain > 0.75):
            st.success(f"🌧️ High rain probability ({prob_rain:.1%})")
        else:
            st.success(f"☀️ Likely sunny ({1-prob_rain:.1%} dry probability)")
            
        st.progress(prob_rain)
        st.caption(f"Model confidence: {prob_rain:.1%} rain probability")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {str(e)}")
        st.info("Please check your inputs and try again")


