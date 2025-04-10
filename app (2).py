import streamlit as st
import pandas as pd
import datetime
from prophet import Prophet
from datetime import timedelta
import requests

# ========== CONFIG ==========
GOOGLE_API_KEY = "******"  # Replace this with your Google API Key
PLACE_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# ========== LOAD DATA ==========
data = pd.read_csv('/content/flight_data_large_cleaned (1).csv')
data.columns = data.columns.str.strip()
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna()

# ========== EMOJI HELPER ==========
def weather_to_emoji(condition):
    return {
        "Clear": "☀️",
        "Cloudy": "⛅",
        "Sunny": "🌞",
        "Rainy": "🌧️",
        "Snowy": "❄️",
        "Stormy": "⛈️",
        "Foggy": "🌫️"
    }.get(condition, "🌍")

# ========== WEATHER PREDICTOR ==========
def get_weather_info(source, destination, date):
    df = data[(data['StartCountry'] == source) & (data['DestinationCountry'] == destination)].copy()
    if df.empty:
        return "No data", "Not enough data to predict weather."

    df['Month'] = df['Date'].dt.month
    target_month = pd.to_datetime(date).month
    df = df[df['Month'] == target_month]

    if df.empty:
        return "No data", "No historical data for this month."

    weather_counts = df['Weather'].value_counts()
    if not weather_counts.empty:
        predicted_weather = weather_counts.idxmax()
        suggestion = "Weather looks good!" if predicted_weather in ["Clear", "Cloudy", "Sunny"] else "Unfavorable weather. Try alternate date."
        return predicted_weather, suggestion
    else:
        return "No data", "No historical weather data."

# ========== ALTERNATIVE DATES (IMPROVED) ==========
def suggest_alternate_dates(source, destination, base_date):
    alternatives = []
    days_checked = 0
    i = 1

    while len(alternatives) < 5 and days_checked < 14:  # Check up to 14 days ahead, suggest max 5 dates
        new_date = base_date + timedelta(days=i)
        weather, _ = get_weather_info(source, destination, new_date)

        if weather in ["Clear", "Cloudy", "Sunny"]:
            alternatives.append({
                'Date': new_date.strftime('%Y-%m-%d'),
                'Weather': weather,
                'Emoji': weather_to_emoji(weather)
            })

        i += 1
        days_checked += 1

    return pd.DataFrame(alternatives)

# ========== FLIGHT COST PREDICTION ==========
def predict_cost(source, destination, travel_date):
    df = data[(data['StartCountry'] == source) & (data['DestinationCountry'] == destination)].copy()
    if df.empty:
        return "No data", "Not enough data to predict cost."

    df = df[['Date', 'Cost']].copy()
    df = df.rename(columns={'Date': 'ds', 'Cost': 'y'})

    if df.shape[0] < 2:
        return "No data", "Insufficient data for model training."

    model = Prophet()
    model.fit(df)

    future = pd.DataFrame({'ds': [pd.to_datetime(travel_date)]})
    forecast = model.predict(future)
    predicted_cost = round(forecast['yhat'].iloc[0], 2)
    return predicted_cost, "Prediction based on historical trends"

# ========== TOURIST ATTRACTIONS ==========
def get_tourist_places(destination):
    params = {
        "query": f"tourist attractions in {destination}",
        "key": GOOGLE_API_KEY
    }
    response = requests.get(PLACE_SEARCH_URL, params=params)
    results = response.json().get("results", [])

    places = []
    for place in results[:5]:  # Limit to top 5
        name = place.get("name", "Unknown")
        address = place.get("formatted_address", "No address")
        lat = place["geometry"]["location"]["lat"]
        lng = place["geometry"]["location"]["lng"]
        rating = place.get("rating", "N/A")
        places.append({
            "Name": name,
            "Address": address,
            "Rating": rating,
            "Lat": lat,
            "Lng": lng
        })
    return pd.DataFrame(places)

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Smart Travel Planner 🌍", layout="centered")
st.title("✈️ Smart Travel Planner")
st.write("Plan your trip smarter with weather, cost prediction, and tourism insights!")

page = st.sidebar.selectbox("Navigate", ["Home", "Analytics", "About"])

# --- Home Page ---
if page == "Home":
    st.subheader("📅 Plan Your Trip")
    source = st.text_input("Source Country")
    destination = st.text_input("Destination Country")
    date = st.date_input("Travel Date", value=datetime.date.today())

    if st.button("Predict"):
        predicted_cost, cost_note = predict_cost(source, destination, date)
        predicted_weather, suggestion = get_weather_info(source, destination, date)

        if predicted_cost == "No data":
            st.error(cost_note)
        else:
            st.success(f"Predicted Flight Cost: ₹{predicted_cost}")
            st.caption(f"📈 {cost_note}")
            emoji = weather_to_emoji(predicted_weather)
            st.info(f"Predicted Weather: {predicted_weather} {emoji}")
            st.warning(suggestion)

            alt_df = suggest_alternate_dates(source, destination, date)
            if not alt_df.empty:
                st.markdown("### ✅ Alternate Travel Dates with Favorable Weather")
                st.dataframe(alt_df)
            else:
                st.warning("⚠️ No favorable alternate dates found in the next 14 days.")

            # Tourist Info
            st.markdown("### 🗺️ Top Tourist Attractions")
            with st.spinner("Fetching attractions..."):
                tourist_df = get_tourist_places(destination)
                if not tourist_df.empty:
                    st.dataframe(tourist_df[["Name", "Address", "Rating"]])
                    st.map(tourist_df.rename(columns={"Lat": "lat", "Lng": "lon"}))
                else:
                    st.info("No tourist data found.")

    st.markdown("### 📋 Weather Stats in Dataset")
    weather_stats = data['Weather'].value_counts().reset_index()
    weather_stats.columns = ['Weather', 'Count']
    weather_stats['Emoji'] = weather_stats['Weather'].apply(weather_to_emoji)
    st.dataframe(weather_stats)

# --- Analytics Page ---
elif page == "Analytics":
    st.subheader("📊 Analytics Dashboard")

    st.markdown("### ✈️ Average Flight Cost by Airline")
    avg_cost_by_airline = data.groupby('Airline')['Cost'].mean().sort_values()
    st.bar_chart(avg_cost_by_airline)

    st.markdown("### 💺 Fuel Cost Comparison by Seat Class")
    avg_fuel_by_class = data.groupby('SeatClass')['FuelCost'].mean().sort_values()
    st.bar_chart(avg_fuel_by_class)

    st.markdown("### 🌍 Passenger Distribution by Destination")
    dest_data = data.groupby('DestinationCountry')['Passenger'].sum()
    st.dataframe(dest_data.sort_values(ascending=False).head(10))

    st.markdown("### 💰 Top 10 Expensive Flights")
    top_flights = data[['FlightName', 'Airline', 'StartCountry', 'DestinationCountry', 'Cost']]
    top_flights = top_flights.sort_values(by='Cost', ascending=False).head(10)
    st.dataframe(top_flights)

# --- About Page ---
elif page == "About":
    st.subheader("ℹ️ About the Project")
    st.write("""
        Smart Travel Planner helps travelers plan better by:
        - Predicting flight cost using historical data
        - Predicting weather conditions based on seasonal trends
        - Suggesting alternate travel dates with better weather
        - Showing top tourist attractions using Google Places API
        - Visualizing popular airlines, costs, and destinations
    """)
