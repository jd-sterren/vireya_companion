import pandas as pd
import numpy as np
from meteostat import Hourly, Point
from datetime import datetime, timedelta
import ollama, os, re, requests, time
import pytz, urllib3, pyodbc
from dotenv import load_dotenv

def weather_api(env_loc="inc/credentials.env"):
    # Load environment variables and OpenWeather API key
    api_key = os.getenv("WEATHER_API")
    
    # Canton Ohio Coordinates
    lat, lon = 40.799, -81.3784

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

def get_current_weather():
    # Define Canton, OH location
    canton = Point(40.799, -81.3784)

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
    data = Hourly(canton, start=start_dt_utc_naive, end=end_dt_utc_naive)
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

def record_weather_data(start_date_local=None, end_date_local=None, excel_file="resources/weather_data.xlsx"):
    """
    Fetches hourly weather data from Meteostat for Canton, OH.
    - Loads existing data from an Excel file if available.
    - If no start/end date is provided, fetches from the last available timestamp in the file.
    - Converts timestamps for Excel compatibility (removes timezone).
    - Appends new data and removes duplicates.
    - Saves back to Excel.

    Args:
        start_date_local (str, optional): Start date in 'YYYY-MM-DD' format (LOCAL TIME). If None, auto-detects.
        end_date_local (str, optional): End date in 'YYYY-MM-DD' format (LOCAL TIME). If None, uses today's date.
        excel_file (str): Filepath to store the data.

    Returns:
        pd.DataFrame: Updated DataFrame with the new and existing weather data.

    Example:
        Example Usage: Get hourly weather data for date and for excel file to pull last date. It does not need
        to be assigned as a variable.
        weather_df = get_hourly_weather("2021-06-01", "2021-06-30")
        weather_df = get_hourly_weather()
    """

    # Define Canton, OH location
    canton = Point(40.799, -81.3784)

    # Define time zones
    eastern = pytz.timezone("US/Eastern")
    utc = pytz.utc

    # Load existing data if the Excel file exists
    existing_df = pd.DataFrame()  # Default to empty DataFrame if no file exists
    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file, parse_dates=["time_local"])
        existing_df["time_local"] = pd.to_datetime(existing_df["time_local"])

    # Auto-detect start date if not provided (use the last recorded time_local)
    if start_date_local is None and not existing_df.empty:
        last_recorded_time = existing_df["time_local"].max()
        start_date_local = last_recorded_time.strftime("%Y-%m-%d")
    elif start_date_local is None:
        start_date_local = (datetime.now(eastern) - timedelta(days=30)).strftime("%Y-%m-%d")  # Default to last 30 days

    # If no end date provided, use the current date
    if end_date_local is None:
        end_date_local = datetime.now(eastern).strftime("%Y-%m-%d")

    # Convert input local dates to datetime objects (with ET timezone)
    start_dt_local = eastern.localize(datetime.strptime(start_date_local, "%Y-%m-%d"), is_dst=None)
    end_dt_local = eastern.localize(datetime.strptime(end_date_local, "%Y-%m-%d"), is_dst=None) + timedelta(days=1)

    # Convert local time to UTC for Meteostat request
    start_dt_utc_naive = start_dt_local.astimezone(utc).replace(tzinfo=None)
    end_dt_utc_naive = end_dt_local.astimezone(utc).replace(tzinfo=None)

    # Fetch hourly weather data from Meteostat in UTC
    data = Hourly(canton, start=start_dt_utc_naive, end=end_dt_utc_naive)
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

    # Create 'weather_relationship' field (rounded local time for merging)
    # df["weather_relationship"] = df["time_local"].dt.round("h")
    # Handle rounding error due to ambiguous times (DST transitions)
    try:
        df["weather_relationship"] = df["time_local"].dt.round("h")
    except pytz.AmbiguousTimeError:
        # If AmbiguousTimeError occurs, try setting ambiguous times to NaT
        df["time_local"] = df["time_local"].dt.tz_localize(None)  # Remove timezone first
        df["weather_relationship"] = df["time_local"].dt.round("h")

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
        "temp": "temp_f",
        "dwpt": "dwpt_f",
        "wspd": "wspd_mph",
        "pres": "pres_inHg",
        "prcp": "prcp_in",
        "snow": "snow_in"
    }, inplace=True)

    # Select relevant columns for merging
    df = df[["time_local", "temp_f", "dwpt_f", "rhum", "prcp_in", "wspd_mph", "wdir", "pres_inHg", "snow_in"]]

    # **Merge with existing data and remove duplicates**
    if not existing_df.empty:
        # Ensure both DataFrames have 'time_local' as a datetime object
        df["time_local"] = pd.to_datetime(df["time_local"])
        existing_df["time_local"] = pd.to_datetime(existing_df["time_local"])

        # Merge, remove duplicates, and sort
        combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(subset=["time_local"]).sort_values("time_local")
    else:
        combined_df = df

    # Ensure 'time_local' is timezone-unaware for Excel compatibility
    combined_df["time_local"] = pd.to_datetime(combined_df["time_local"]).dt.tz_localize(None)
    combined_df["weather_relationship"] = pd.to_datetime(combined_df["time_local"]).dt.tz_localize(None)

    # **Save the updated DataFrame to Excel**
    combined_df.to_excel(excel_file, index=False)

    return combined_df
