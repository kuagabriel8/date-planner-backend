import httpx
import respx

from tools import (
    NEA_2HR_FORECAST_URL,
    NEA_4DAY_FORECAST_URL,
    NEA_24HR_FORECAST_URL,
    NEA_PSI_URL,
    NEA_UV_INDEX_URL,
    GOOGLE_PLACES_TEXT_SEARCH_URL,
    check_weather,
    check_psi,
    check_uv_index,
    find_places,
)

# --- Fixtures: minimal real-shaped payloads (trimmed from live responses) ---

TWO_HR_PAYLOAD = {
    "data": {
        "items": [
            {
                "valid_period": {"text": "7.30 pm to 9.30 pm"},
                "forecasts": [
                    {"area": "Bishan", "forecast": "Fair (Night)"},
                    {"area": "Orchard", "forecast": "Cloudy"},
                ],
            }
        ]
    }
}

TWENTY_FOUR_HR_PAYLOAD = {
    "data": {
        "records": [
            {
                "general": {
                    "validPeriod": {"text": "6 PM 12 Aug to 6 PM 13 Aug"},
                    "forecast": {"text": "Fair (Day)"},
                    "temperature": {"low": 26, "high": 34},
                    "relativeHumidity": {"low": 55, "high": 90},
                    "wind": {"direction": "SSE", "speed": {"low": 10, "high": 20}},
                },
                "periods": [
                    {
                        "timePeriod": {"text": "6 pm 12 Aug to 6 am 13 Aug"},
                        "regions": {"central": {"text": "Fair (Night)"}},
                    }
                ],
            }
        ]
    }
}

FOUR_DAY_PAYLOAD = {
    "data": {
        "records": [
            {
                "forecasts": [
                    {
                        "day": "Thursday",
                        "forecast": {"text": "Fair (Day)"},
                        "temperature": {"low": 26, "high": 34},
                        "relativeHumidity": {"low": 55, "high": 90},
                    }
                ]
            }
        ]
    }
}

PSI_PAYLOAD = {
    "data": {
        "items": [
            {
                "readings": {
                    "psi_twenty_four_hourly": {
                        "north": 57,
                        "south": 59,
                        "east": 67,
                        "west": 65,
                        "central": 155,
                    }
                }
            }
        ]
    }
}

UV_PAYLOAD = {
    "data": {
        "records": [
            {
                "index": [
                    {"hour": "2026-08-12T19:00:00+08:00", "value": 0},
                    {"hour": "2026-08-12T14:00:00+08:00", "value": 9},
                    {"hour": "2026-08-12T13:00:00+08:00", "value": 5},
                ]
            }
        ]
    }
}


class TestCheckWeatherHorizonSelection:
    @respx.mock
    async def test_default_hours_ahead_uses_2hr_only(self):
        respx.mock.get(NEA_2HR_FORECAST_URL).mock(return_value=httpx.Response(200, json=TWO_HR_PAYLOAD))
        route_24hr = respx.mock.get(NEA_24HR_FORECAST_URL)
        route_4day = respx.mock.get(NEA_4DAY_FORECAST_URL)

        result = await check_weather.ainvoke({})

        assert "2-hour forecast" in result
        assert not route_24hr.called
        assert not route_4day.called

    @respx.mock
    async def test_hours_ahead_boundary_2_uses_2hr(self):
        respx.mock.get(NEA_2HR_FORECAST_URL).mock(return_value=httpx.Response(200, json=TWO_HR_PAYLOAD))

        result = await check_weather.ainvoke({"hours_ahead": 2})

        assert "2-hour forecast" in result

    @respx.mock
    async def test_hours_ahead_3_uses_24hr_only(self):
        route_2hr = respx.mock.get(NEA_2HR_FORECAST_URL)
        respx.mock.get(NEA_24HR_FORECAST_URL).mock(
            return_value=httpx.Response(200, json=TWENTY_FOUR_HR_PAYLOAD)
        )

        result = await check_weather.ainvoke({"hours_ahead": 3})

        assert "24-hour forecast" in result
        assert "central: Fair (Night)" in result
        assert not route_2hr.called

    @respx.mock
    async def test_hours_ahead_boundary_24_uses_24hr(self):
        respx.mock.get(NEA_24HR_FORECAST_URL).mock(
            return_value=httpx.Response(200, json=TWENTY_FOUR_HR_PAYLOAD)
        )

        result = await check_weather.ainvoke({"hours_ahead": 24})

        assert "24-hour forecast" in result

    @respx.mock
    async def test_hours_ahead_25_uses_4day_only(self):
        route_24hr = respx.mock.get(NEA_24HR_FORECAST_URL)
        respx.mock.get(NEA_4DAY_FORECAST_URL).mock(
            return_value=httpx.Response(200, json=FOUR_DAY_PAYLOAD)
        )

        result = await check_weather.ainvoke({"hours_ahead": 25})

        assert "4-day outlook" in result
        assert "Thursday: Fair (Day)" in result
        assert not route_24hr.called

    @respx.mock
    async def test_far_future_still_uses_4day(self):
        respx.mock.get(NEA_4DAY_FORECAST_URL).mock(
            return_value=httpx.Response(200, json=FOUR_DAY_PAYLOAD)
        )

        result = await check_weather.ainvoke({"hours_ahead": 96})

        assert "4-day outlook" in result


class TestCheckWeatherFormatting:
    @respx.mock
    async def test_area_filter_matches(self):
        respx.mock.get(NEA_2HR_FORECAST_URL).mock(return_value=httpx.Response(200, json=TWO_HR_PAYLOAD))

        result = await check_weather.ainvoke({"area": "bishan"})

        assert "Bishan: Fair (Night)" in result
        assert "Orchard" not in result

    @respx.mock
    async def test_area_filter_no_match(self):
        respx.mock.get(NEA_2HR_FORECAST_URL).mock(return_value=httpx.Response(200, json=TWO_HR_PAYLOAD))

        result = await check_weather.ainvoke({"area": "Nonexistent Place"})

        assert "No forecast area matched" in result

    @respx.mock
    async def test_no_area_returns_all(self):
        respx.mock.get(NEA_2HR_FORECAST_URL).mock(return_value=httpx.Response(200, json=TWO_HR_PAYLOAD))

        result = await check_weather.ainvoke({})

        assert "Bishan: Fair (Night)" in result
        assert "Orchard: Cloudy" in result


class TestCheckWeatherErrors:
    @respx.mock
    async def test_http_error_returns_message_not_raises(self):
        respx.mock.get(NEA_2HR_FORECAST_URL).mock(return_value=httpx.Response(500))

        result = await check_weather.ainvoke({})

        assert "Could not fetch weather data" in result

    @respx.mock
    async def test_network_error_returns_message_not_raises(self):
        respx.mock.get(NEA_2HR_FORECAST_URL).mock(side_effect=httpx.ConnectError("boom"))

        result = await check_weather.ainvoke({})

        assert "Could not fetch weather data" in result


class TestCheckPsi:
    @respx.mock
    async def test_all_regions(self):
        respx.mock.get(NEA_PSI_URL).mock(return_value=httpx.Response(200, json=PSI_PAYLOAD))

        result = await check_psi.ainvoke({})

        assert "north: 57 (Moderate)" in result
        assert "central: 155 (Unhealthy)" in result

    @respx.mock
    async def test_single_region(self):
        respx.mock.get(NEA_PSI_URL).mock(return_value=httpx.Response(200, json=PSI_PAYLOAD))

        result = await check_psi.ainvoke({"region": "Central"})

        assert result == "PSI (24-hr, central): 155 (Unhealthy)"

    @respx.mock
    async def test_invalid_region(self):
        respx.mock.get(NEA_PSI_URL).mock(return_value=httpx.Response(200, json=PSI_PAYLOAD))

        result = await check_psi.ainvoke({"region": "nowhere"})

        assert "No PSI region matched" in result

    @respx.mock
    async def test_http_error_returns_message_not_raises(self):
        respx.mock.get(NEA_PSI_URL).mock(return_value=httpx.Response(500))

        result = await check_psi.ainvoke({})

        assert "Could not fetch PSI data" in result


class TestCheckUvIndex:
    @respx.mock
    async def test_reports_current_and_peak(self):
        respx.mock.get(NEA_UV_INDEX_URL).mock(return_value=httpx.Response(200, json=UV_PAYLOAD))

        result = await check_uv_index.ainvoke({})

        assert "Current UV index: 0 (Low)" in result
        assert "Today's peak: 9 (Very High)" in result

    @respx.mock
    async def test_http_error_returns_message_not_raises(self):
        respx.mock.get(NEA_UV_INDEX_URL).mock(return_value=httpx.Response(500))

        result = await check_uv_index.ainvoke({})

        assert "Could not fetch UV index data" in result


class TestFindPlaces:
    async def test_missing_api_key_returns_message(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

        result = await find_places.ainvoke({"query": "ramen"})

        assert "not configured" in result

    @respx.mock
    async def test_success_formats_places(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key")
        respx.mock.post(GOOGLE_PLACES_TEXT_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "places": [
                        {
                            "displayName": {"text": "Ramen Place"},
                            "formattedAddress": "123 Orchard Rd",
                            "rating": 4.5,
                            "userRatingCount": 200,
                            "googleMapsUri": "https://maps.google.com/?cid=1",
                        }
                    ]
                },
            )
        )

        result = await find_places.ainvoke({"query": "ramen"})

        assert "Ramen Place" in result
        assert "4.5/5 (200 reviews)" in result
        assert "123 Orchard Rd" in result

    @respx.mock
    async def test_no_results(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key")
        respx.mock.post(GOOGLE_PLACES_TEXT_SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"places": []})
        )

        result = await find_places.ainvoke({"query": "nonexistent cuisine xyz"})

        assert "No places found" in result

    @respx.mock
    async def test_http_error_returns_message_not_raises(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key")
        respx.mock.post(GOOGLE_PLACES_TEXT_SEARCH_URL).mock(return_value=httpx.Response(403))

        result = await find_places.ainvoke({"query": "ramen"})

        assert "Could not search places" in result
