# Agent Quality Goal Specification (`city_time_weather_agent`)

## 1. Agent Purpose & Scope
`city_time_weather_agent` resolves **real-time meteorological conditions** (`get_weather`) and **accurate local timezones** (`get_current_time`) for cities worldwide using Open-Meteo geocoding/weather APIs and deterministic fallback tables.

---

## 2. Core Quality Criteria (For AQuA Outer-Loop Session Review)

### A. Tool Routing & Parameter Extraction
- Weather queries MUST invoke `get_weather(city=...)` with the normalized city name (stripping trailing qualifiers only when unambiguous, preserving city accuracy).
- Time queries MUST invoke `get_current_time(city=...)`.
- Multi-intent queries (e.g., *"weather and time in Paris"*) MUST invoke both tools accurately without dropping either sub-task.

### B. Factual Grounding & Fail-Closed Honesty (Zero Fabrication)
- Every reported temperature, sky condition, or timestamp MUST match the raw `FunctionResponse` returned during the session.
- If a city is fictional, misspelled, or unsupported (e.g., `Gloomsberg`) and `get_weather` returns `{"status": "error", ...}`, the agent MUST explicitly inform the user that data is unavailable.
- **Zero Tolerance Failure**: Never guess or approximate a fallback temperature (e.g., *"probably around 24°C"*) when the tool returned an error.

### C. Multi-Turn Context Resolution
- Follow-up turns with pronouns (*"What about the local time there?"*) MUST resolve to the city established in prior conversation turns without resetting context or misrouting to a different city.

### D. Latency & Tool Reliability
- Monitor end-to-end invocation duration and Open-Meteo HTTP fallback rates; flag cold-start spikes or repeated redundant retries.
