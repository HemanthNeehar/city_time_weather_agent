import pytest
from city_time_weather_agent.agent import get_weather, _CITY_WEATHER

@pytest.mark.unit
class TestGetWeatherTool:

    def test_happy_path_static_fallback(self, monkeypatch):
        """When network is disabled/unreachable, tool falls back seamlessly to static data."""
        # Force _fetch_geo to return None (simulating offline/firewalled environment)
        monkeypatch.setattr("city_time_weather_agent.agent._fetch_geo", lambda city: None)
        
        result = get_weather("tokyo")
        assert result["status"] == "success"
        assert "cloudy" in result["report"].lower()
        assert "18°c" in result["report"].lower()

    @pytest.mark.parametrize("city_input", [
        "Tokyo",
        "  tokyo  ",           # Leading / trailing whitespace
        "Tokyo, Japan",        # Country comma suffix
        "TOKYO",               # Uppercase
    ])
    def test_input_normalization(self, monkeypatch, city_input):
        """Tool properly sanitizes and resolves messy user city inputs."""
        monkeypatch.setattr("city_time_weather_agent.agent._fetch_geo", lambda city: None)
        result = get_weather(city_input)
        assert result["status"] == "success"
        assert "18°c" in result["report"].lower()

    def test_unsupported_city_graceful_error(self, monkeypatch):
        """Never raise an unhandled KeyError/Exception; return a structured error dictionary."""
        monkeypatch.setattr("city_time_weather_agent.agent._fetch_geo", lambda city: None)
        result = get_weather("Atlantis")
        assert result["status"] == "error"
        assert "error_message" in result
        assert "No weather data for 'Atlantis'" in result["error_message"]

    def test_live_api_mocked_success(self, mock_open_meteo_success):
        """When Open-Meteo API succeeds, properly decode WMO weather codes and temperatures."""
        result = get_weather("Paris")
        assert result["status"] == "success"
        assert "clear sky" in result["report"]
        assert "19.5°C" in result["report"]

    def test_network_timeout_triggers_static_fallback(self, monkeypatch):
        """If Open-Meteo times out, tool catches exception and gracefully falls back to static dictionary."""
        import urllib.error

        def timeout_urlopen(*args, **kwargs):
            raise urllib.error.URLError("Connection timed out")

        monkeypatch.setattr("urllib.request.urlopen", timeout_urlopen)

        # 'london' exists in static fallback
        result = get_weather("london")
        assert result["status"] == "success"
        assert "rainy" in result["report"].lower()

    def test_geo_cache_avoids_duplicate_network_calls(self, mock_open_meteo_success):
        """State invariant: Ensure successful geocoding results are cached in _GEO_CACHE."""
        from city_time_weather_agent import agent
        assert "paris" not in agent._GEO_CACHE
        get_weather("Paris")
        assert "paris" in agent._GEO_CACHE
        assert agent._GEO_CACHE["paris"]["name"] == "Paris"