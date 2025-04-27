import pandas as pd
import numpy as np
from meteostat import Hourly, Point
from datetime import datetime, timedelta
import ollama, os, re, requests, time
import pytz, urllib3, pyodbc
from dotenv import load_dotenv

def weather_api(lat=40.799, lon=-81.3784):
    # Load environment variables and OpenWeather API key
    api_key = os.getenv("WEATHER_API")

    # URL Setup
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=imperial"

    # Call API
    response = requests.get(url).json()

    # Format data to save as DataFrame
    flattened_data = {
        'longitude': response['coord']['lon'],
        'latitude': response['coord']['lat'],
        'weather_id': response['weather'][0]['id'],
        'weather_main': response['weather'][0]['main'],
        'weather_description': response['weather'][0]['description'],
        'temperature': response['main']['temp'],
        'feels_like': response['main']['feels_like'],
        'temp_min': response['main']['temp_min'],
        'temp_max': response['main']['temp_max'],
        'pressure': response['main']['pressure'],
        'humidity': response['main']['humidity'],
        'visibility': response.get('visibility', None),  # Default to None if not available
        'wind_speed': response['wind'].get('speed', None),
        'wind_deg': response['wind'].get('deg', None),
        'wind_gust': response['wind'].get('gust', None),  # Handle missing gust
        'cloud_coverage': response['clouds'].get('all', None),
        'timestamp': datetime.fromtimestamp(response['dt']),  # Convert from Unix
        'country': response['sys'].get('country', None),
        'sunrise': datetime.fromtimestamp(response['sys']['sunrise']) if 'sunrise' in response['sys'] else None,
        'sunset': datetime.fromtimestamp(response['sys']['sunset']) if 'sunset' in response['sys'] else None,
        'city_name': response.get('name', None)
    }

    return flattened_data

def load_context(CONTEXT_FILE = "inc/logs/vireya_context.txt"):
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def save_context(reflection, CONTEXT_FILE = "inc/logs/vireya_context.txt"):
    os.makedirs(os.path.dirname(CONTEXT_FILE), exist_ok=True)
    with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
        f.write(reflection.strip())

def parse_context_timestamp_and_body(context_text):
    if context_text.startswith("["):
        try:
            timestamp_end = context_text.index("]")
            timestamp = context_text[1:timestamp_end]
            summary = context_text[timestamp_end + 1:].strip()
            return timestamp, summary
        except ValueError:
            pass  # fallback below
    return None, context_text.strip()

## ----------------- Date Operations Functions ----------------- ##
def today(fmt=None):
    return _format_date(datetime.now(), fmt)

def yesterday(fmt=None):
    return _format_date(today() - timedelta(days=1), fmt)

def last_week(fmt=None):
    return _format_date(today() - timedelta(days=7), fmt)

def last_month(fmt=None):
    return _format_date(today() - timedelta(days=30), fmt)

def last_quarter(fmt=None):
    return _format_date(today() - timedelta(days=90), fmt)

def _format_date(date_obj, fmt):
    """Helper function to format the date if needed."""
    return date_obj.strftime("%Y-%m-%d") if fmt == "str" else date_obj

def get_current_datetime(fmt=None):
    return datetime.now().strftime("%A, %B %d, %Y %H:%M:%S") if fmt == "str" else datetime.now()

def get_current_weather(lat=40.799, lon=-81.3784):
    # Define Canton, OH location
    location = Point(lat, lon)

    # Define time zones
    eastern = pytz.timezone("US/Eastern")
    utc = pytz.utc

    # Convert input local dates to datetime objects (with ET timezone)
    start_dt_local = eastern.localize(datetime.strptime(datetime.now(eastern).strftime("%Y-%m-%d"), "%Y-%m-%d"), is_dst=None)
    end_dt_local = eastern.localize(datetime.strptime(datetime.now(eastern).strftime("%Y-%m-%d"), "%Y-%m-%d"), is_dst=None) + timedelta(days=1)

    # Convert local time to UTC for Meteostat request
    start_dt_utc_naive = start_dt_local.astimezone(utc).replace(tzinfo=None)
    end_dt_utc_naive = end_dt_local.astimezone(utc).replace(tzinfo=None)

    # Fetch hourly weather data from Meteostat in UTC
    data = Hourly(location, start=start_dt_utc_naive, end=end_dt_utc_naive)
    df = data.fetch()

    # Reset index to access 'time' as a column
    df.reset_index(inplace=True)

    # Convert 'time' column explicitly to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Handle AmbiguousTimeError during DST changes
    try:
        df["time_local"] = df["time"].dt.tz_localize(utc).dt.tz_convert(eastern)
    except pytz.AmbiguousTimeError:
        df["time_local"] = df["time"].dt.tz_localize(utc).dt.tz_convert(eastern, ambiguous="NaT")

    # Remove rows with NaT (caused by ambiguous times that cannot be resolved)
    df = df.dropna(subset=["time_local"])

    # Filter out any future timestamps
    current_utc_time = datetime.utcnow().replace(tzinfo=None)
    df = df[df["time"] <= current_utc_time]

    # Convert temperature & dew point from Celsius to Fahrenheit
    df["temp"] = df["temp"] * 9/5 + 32
    df["dwpt"] = df["dwpt"] * 9/5 + 32

    # Convert wind speed from km/h to mph
    df["wspd"] = df["wspd"] * 0.621371

    # Convert pressure from hPa to inHg
    df["pres"] = df["pres"] * 0.02953

    # Convert precipitation from mm to inches
    df["prcp"] = df["prcp"] * 0.0393701

    # Convert snow depth from cm to inches
    df["snow"] = df["snow"] * 0.393701

    # Remove timezone information for Excel compatibility
    df["time_local"] = df["time_local"].dt.strftime("%Y-%m-%d %H:%M:%S")  # Removes timezone offset

    # Rename columns for clarity
    df.rename(columns={
        "time_local": "DateTime_Recorded",
        "temp": "Temperature_F",
        "rhum": "Relative_Humidity_%",
        "dwpt": "Dew_Point_F",
        "wspd": "Wind_Speed_mph",
        "pres": "Pressure_inHg",
        "prcp": "Precipitation_in",
        "snow": "snow_in",
        "wdir": "Wind_Direction_deg",
    }, inplace=True)

    # Select relevant columns for merging
    df = df[["DateTime_Recorded", "Temperature_F", "Dew_Point_F", "Relative_Humidity_%", "Precipitation_in", "Wind_Speed_mph", "Wind_Direction_deg", "Pressure_inHg", "snow_in"]]

    return df.iloc[-1]  # Return the most recent row of data
