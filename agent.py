# city_time_weather_agent/agent.py
import json
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import LlmAgent

# Use Windows/system certificate store if available (avoids SSL issues behind proxies)
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

# Fallback static timezone data
_CITY_TZ = {
    "new york": "America/New_York",
    "tokyo": "Asia/Tokyo",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "delhi": "Asia/Kolkata",
    "sydney": "Australia/Sydney",
    "colombo": "Asia/Colombo",
    "berlin": "Europe/Berlin",
    "netherlands": "Europe/Amsterdam",
    "amsterdam": "Europe/Amsterdam",
    "chennai": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata",
    "bengaluru": "Asia/Kolkata",
    "hyderabad": "Asia/Kolkata",
    "italy": "Europe/Rome",
    "rome": "Europe/Rome",
    "dubai": "Asia/Dubai",
    "singapore": "Asia/Singapore",
    "san francisco": "America/Los_Angeles",
    "los angeles": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "toronto": "America/Toronto",
}

# Fallback static weather data
_CITY_WEATHER = {
    "new york": ("sunny", 25),
    "tokyo": ("cloudy", 18),
    "london": ("rainy", 14),
    "paris": ("sunny", 20),
    "delhi": ("hot", 35),
    "sydney": ("sunny", 22),
    "colombo": ("sunny", 30),
    "berlin": ("cloudy", 16),
    "netherlands": ("cloudy", 15),
    "amsterdam": ("cloudy", 15),
    "chennai": ("hot", 34),
    "mumbai": ("hot", 33),
    "bangalore": ("hot", 32),
    "bengaluru": ("hot", 32),
    "hyderabad": ("hot", 34),
    "italy": ("sunny", 21),
    "rome": ("sunny", 21),
    "dubai": ("hot", 40),
    "singapore": ("humid", 31),
    "san francisco": ("breezy", 17),
    "los angeles": ("sunny", 24),
    "chicago": ("windy", 15),
    "toronto": ("cool", 12),
}

# WMO Weather interpretation codes (Open-Meteo)
_WMO_WEATHER_MAP = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

_GEO_CACHE: dict[str, dict] = {}

def _fetch_geo(city: str) -> dict | None:
    """Fetches geographical coordinates and timezone for a city using Open-Meteo Geocoding API (free, no API key)."""
    clean = city.strip()
    key = clean.lower()
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]

    queries = [clean]
    if "," in clean:
        queries.append(clean.split(",")[0].strip())

    for q in queries:
        try:
            encoded = urllib.parse.quote(q)
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1&language=en&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "CityTimeWeatherAgent/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results")
                if results:
                    info = results[0]
                    res = {
                        "name": info.get("name", clean),
                        "country": info.get("country", ""),
                        "latitude": info.get("latitude"),
                        "longitude": info.get("longitude"),
                        "timezone": info.get("timezone"),
                    }
                    _GEO_CACHE[key] = res
                    return res
        except Exception:
            pass
    return None

def get_weather(city: str) -> dict:
    """Returns the current weather condition and temperature (Celsius) for a city."""
    key = city.lower().strip()
    base_key = key.split(",")[0].strip() if "," in key else key
    display_name = city.strip()

    # Attempt real-world fetch from Open-Meteo (free, no API key required)
    geo = _fetch_geo(city)
    if geo and geo.get("latitude") is not None and geo.get("longitude") is not None:
        lat = geo["latitude"]
        lon = geo["longitude"]
        display_name = geo.get("name", display_name)
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code"
            req = urllib.request.Request(url, headers={"User-Agent": "CityTimeWeatherAgent/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                current = data.get("current", {})
                temp_c = current.get("temperature_2m")
                code = current.get("weather_code")
                condition = _WMO_WEATHER_MAP.get(code, "clear")
                if temp_c is not None:
                    return {
                        "status": "success",
                        "report": f"The weather in {display_name} is {condition} with a temperature of {temp_c}°C.",
                    }
        except Exception:
            pass

    # Fallback to static data
    weather_entry = _CITY_WEATHER.get(key) or _CITY_WEATHER.get(base_key)
    if weather_entry:
        condition, temp_c = weather_entry
        return {
            "status": "success",
            "report": f"The weather in {display_name} is {condition} with a temperature of {temp_c}°C.",
        }

    return {"status": "error", "error_message": f"No weather data for '{city}'."}

def get_current_time(city: str) -> dict:
    """Returns the current local time for a city."""
    key = city.lower().strip()
    base_key = key.split(",")[0].strip() if "," in key else key

    tz_name = None
    display_name = city.strip()

    # Try resolving timezone dynamically via Open-Meteo geocoding
    geo = _fetch_geo(city)
    if geo and geo.get("timezone"):
        tz_name = geo["timezone"]
        display_name = geo.get("name", display_name)
    elif key in _CITY_TZ:
        tz_name = _CITY_TZ[key]
    elif base_key in _CITY_TZ:
        tz_name = _CITY_TZ[base_key]

    if not tz_name:
        return {"status": "error", "error_message": f"No timezone data for '{city}'."}

    try:
        now = datetime.now(ZoneInfo(tz_name))
        return {
            "status": "success",
            "report": f"The current time in {display_name} is {now.strftime('%Y-%m-%d %H:%M:%S %Z')}.",
        }
    except Exception as e:
        return {"status": "error", "error_message": f"Error calculating time for '{city}': {str(e)}"}

root_agent = LlmAgent(
    name="city_time_weather_agent",
    model="gemini-3.5-flash",
    description="Answers questions about the current weather and local time in a city.",
    instruction=(
        "You are a helpful assistant. Use the `get_weather` tool for weather questions and the "
        "`get_current_time` tool for time questions. Always call a tool before answering; never guess."
    ),
    tools=[get_weather, get_current_time],
)
