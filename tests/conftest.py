# tests/conftest.py
import sys
from pathlib import Path

# Add project root and its parent directory to sys.path so that both
# `import agent` and `from city_time_weather_agent import agent` resolve
# regardless of whether pytest is run from this directory, the parent directory, or an IDE.
_ROOT = Path(__file__).resolve().parent.parent
_PARENT = _ROOT.parent
for _p in (str(_ROOT), str(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from city_time_weather_agent import agent

@pytest.fixture(autouse=True)
def reset_geo_cache():
    """Clear the in-memory geocoding cache before each test."""
    agent._GEO_CACHE.clear()
    yield
    agent._GEO_CACHE.clear()

@pytest.fixture
def mock_open_meteo_success(monkeypatch):
    """Mocks urllib.request.urlopen to return deterministic Open-Meteo JSON responses."""
    import io
    import json

    class MockResponse:
        def __init__(self, data):
            self.data = json.dumps(data).encode("utf-8")
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def custom_urlopen(req, timeout=5):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "geocoding-api.open-meteo.com" in url:
            return MockResponse({
                "results": [{
                    "name": "Paris",
                    "country": "France",
                    "latitude": 48.8534,
                    "longitude": 2.3488,
                    "timezone": "Europe/Paris"
                }]
            })
        elif "api.open-meteo.com/v1/forecast" in url:
            return MockResponse({
                "current": {
                    "temperature_2m": 19.5,
                    "weather_code": 0  # clear sky
                }
            })
        raise ValueError(f"Unhandled mock URL: {url}")

    monkeypatch.setattr("urllib.request.urlopen", custom_urlopen)