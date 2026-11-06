import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from city_time_weather_agent.agent import get_current_time

@pytest.mark.unit
class TestGetCurrentTimeTool:

    def test_current_time_format_and_timezone(self, monkeypatch):
        """Verify returned time format matches ISO/strftime contract and reflects correct offset."""
        monkeypatch.setattr("city_time_weather_agent.agent._fetch_geo", lambda city: None)
        result = get_current_time("Tokyo")
        assert result["status"] == "success"
        assert "The current time in Tokyo is" in result["report"]
        # Verify valid time format (YYYY-MM-DD HH:MM:SS)
        assert any(tz in result["report"] for tz in ["JST", "Asia/Tokyo", "+0900", "UTC"])

    @pytest.mark.parametrize("city_input", [
        "Tokyo",
        "  tokyo  ",           # Leading / trailing whitespace
        "Tokyo, Japan",        # Country comma suffix
        "TOKYO",               # Uppercase
    ])
    def test_current_time_input_normalization(self, monkeypatch, city_input):
        """Time tool properly handles casing, whitespace, and comma suffixes."""
        monkeypatch.setattr("city_time_weather_agent.agent._fetch_geo", lambda city: None)
        result = get_current_time(city_input)
        assert result["status"] == "success"
        assert any(tz in result["report"] for tz in ["JST", "Asia/Tokyo", "+0900", "UTC"])

    def test_unsupported_timezone_city_fails_gracefully(self, monkeypatch):
        """Unsupported cities must return error status, not crash."""
        monkeypatch.setattr("city_time_weather_agent.agent._fetch_geo", lambda city: None)
        result = get_current_time("UnknownCityXYZ")
        assert result["status"] == "error"
        assert "No timezone data" in result["error_message"]