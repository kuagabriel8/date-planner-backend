import os

import httpx
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# Singapore government real-time environment APIs (data.gov.sg). No API key
# required. Verified live against api-open.data.gov.sg/v2 on 2026-08-12.
NEA_2HR_FORECAST_URL = "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast"
NEA_24HR_FORECAST_URL = "https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast"
NEA_4DAY_FORECAST_URL = "https://api-open.data.gov.sg/v2/real-time/api/four-day-outlook"
NEA_PSI_URL = "https://api-open.data.gov.sg/v2/real-time/api/psi"
NEA_UV_INDEX_URL = "https://api-open.data.gov.sg/v2/real-time/api/uv"
#theres more APIs but some are useless and i havent tested the rest

# Places API (New) - Text Search.
GOOGLE_PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"


async def _fetch_json(client: httpx.AsyncClient, url: str) -> dict:
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.json()["data"]


def _format_2hr(area: str, forecast_data) -> str:
    latest = forecast_data["items"][0]
    valid_period = latest["valid_period"]["text"]
    forecasts = latest["forecasts"]

    if area:
        matches = [f for f in forecasts if area.lower() in f["area"].lower()]
        if not matches:
            return (
                f"No forecast area matched '{area}'. Try a Singapore planning "
                "area name like 'Bishan', 'Orchard', or 'Sentosa'."
            )
        forecast_lines = [f"{f['area']}: {f['forecast']}" for f in matches]
    else:
        forecast_lines = [f"{f['area']}: {f['forecast']}" for f in forecasts]

    return "\n".join([f"2-hour forecast ({valid_period}):", *forecast_lines])


def _format_24hr(forecast_data) -> str:
    record = forecast_data["records"][0]
    general = record["general"]
    lines = [
        f"24-hour forecast ({general['validPeriod']['text']}):",
        f"{general['forecast']['text']}, "
        f"{general['temperature']['low']}-{general['temperature']['high']}°C, "
        f"humidity {general['relativeHumidity']['low']}-{general['relativeHumidity']['high']}%, "
        f"wind {general['wind']['direction']} {general['wind']['speed']['low']}-{general['wind']['speed']['high']} km/h",
        "",
        "By region and time period:",
    ]
    for period in record["periods"]:
        regions = ", ".join(f"{name}: {r['text']}" for name, r in period["regions"].items())
        lines.append(f"{period['timePeriod']['text']} - {regions}")
    return "\n".join(lines)


def _format_4day(forecast_data) -> str:
    record = forecast_data["records"][0]
    lines = ["4-day outlook:"]
    for day in record["forecasts"]:
        lines.append(
            f"{day['day']}: {day['forecast']['text']}, "
            f"{day['temperature']['low']}-{day['temperature']['high']}°C, "
            f"humidity {day['relativeHumidity']['low']}-{day['relativeHumidity']['high']}%"
        )
    return "\n".join(lines)


@tool
async def check_weather(area: str = "", hours_ahead: int = 0) -> str:
    """Get Singapore's weather/rain/temperature forecast. ALWAYS use this tool
    (never web_search) for any question about weather, rain, or forecast in
    Singapore - it returns live official NEA data.

    Getting hours_ahead right matters: it picks exactly one NEA forecast to
    call, so set it to match how far out the date/activity actually is, not
    a default guess:
    - hours_ahead 0-2 (default): 2-hour forecast, granular by planning area.
      Use for "right now" / "in the next hour or two" questions.
    - hours_ahead 3-24: 24-hour forecast, by region only (no planning areas).
      Use for "later today" / "tonight" / "tomorrow morning" questions.
    - hours_ahead 25+ (up to 4 days / 96h): 4-day islandwide outlook.
      Use for "this weekend" / "in 3 days" questions.
    Passing too small a value silently loses information the date needs;
    passing too large a value silently loses area-level granularity.

    Args:
        area: A Singapore planning area name to filter the forecast for
            (e.g. "Bishan", "Orchard", "Sentosa"). Only used for the 0-2h
            forecast; ignored beyond that since NEA only publishes region-
            or island-level detail further out.
        hours_ahead: How many hours from now the date/activity is planned
            for. Use 0 (default) for right now.
    """
    if hours_ahead <= 2:
        url, formatter = NEA_2HR_FORECAST_URL, lambda data: _format_2hr(area, data)
    elif hours_ahead <= 24:
        url, formatter = NEA_24HR_FORECAST_URL, _format_24hr
    else:
        url, formatter = NEA_4DAY_FORECAST_URL, _format_4day

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            data = await _fetch_json(client, url)
    except httpx.HTTPError as e:
        return f"Could not fetch weather data: {e}"

    return formatter(data)


def _psi_category(value: int) -> str:
    if value <= 50:
        return "Good"
    elif value <= 100:
        return "Moderate"
    elif value <= 200:
        return "Unhealthy"
    elif value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def _format_psi(data, region: str) -> str:
    readings = data["items"][0]["readings"]["psi_twenty_four_hourly"]

    if region:
        region = region.lower()
        if region not in readings:
            return (
                f"No PSI region matched '{region}'. Valid regions: "
                f"{', '.join(sorted(readings))}."
            )
        value = readings[region]
        return f"PSI (24-hr, {region}): {value} ({_psi_category(value)})"

    lines = ["PSI (24-hr) by region:"]
    lines += [f"{r}: {v} ({_psi_category(v)})" for r, v in readings.items()]
    return "\n".join(lines)


@tool
async def check_psi(region: str = "") -> str:
    """Get Singapore's current PSI (Pollutant Standards Index), 24-hour reading.
    ALWAYS use this tool (never web_search) for any question mentioning PSI,
    haze, smoke, air quality, or air pollution in Singapore - it returns live
    official NEA data that web_search cannot beat.

    Args:
        region: One of "north", "south", "east", "west", "central". Leave
            empty for all regions.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            data = await _fetch_json(client, NEA_PSI_URL)
    except httpx.HTTPError as e:
        return f"Could not fetch PSI data: {e}"

    return _format_psi(data, region)


def _uv_category(value: int) -> str:
    if value <= 2:
        return "Low"
    elif value <= 5:
        return "Moderate"
    elif value <= 7:
        return "High"
    elif value <= 10:
        return "Very High"
    return "Extreme"


def _format_uv(data) -> str:
    readings = data["records"][0]["index"]
    if not readings:
        return "No UV index data available."

    current = max(readings, key=lambda r: r["hour"])
    peak = max(readings, key=lambda r: r["value"])
    return (
        f"Current UV index: {current['value']} ({_uv_category(current['value'])})\n"
        f"Today's peak: {peak['value']} ({_uv_category(peak['value'])}) at {peak['hour']}"
    )


@tool
async def check_uv_index() -> str:
    """Get Singapore's current UV index and today's peak UV index. ALWAYS use
    this tool (never web_search) for any question mentioning UV, UV index,
    sun exposure, or sunburn risk in Singapore - it returns live official NEA
    data that web_search cannot beat.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            data = await _fetch_json(client, NEA_UV_INDEX_URL)
    except httpx.HTTPError as e:
        return f"Could not fetch UV index data: {e}"

    return _format_uv(data)


@tool
async def find_places(query: str, location_bias: str = "Singapore") -> str:
    """Search for real places (restaurants, activities, venues) via Google Maps.

    Args:
        query: What to look for, e.g. "romantic italian restaurant" or
            "rock climbing gym".
        location_bias: Area to narrow results to. Defaults to "Singapore".
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return "Places search is not configured (missing GOOGLE_MAPS_API_KEY)."

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                GOOGLE_PLACES_TEXT_SEARCH_URL,
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": api_key,
                    "X-Goog-FieldMask": (
                        "places.displayName,places.formattedAddress,places.rating,"
                        "places.userRatingCount,places.priceLevel,places.googleMapsUri"
                    ),
                },
                json={"textQuery": f"{query} in {location_bias}", "pageSize": 8},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        return f"Could not search places: {e}"

    places = data.get("places", [])
    if not places:
        return f"No places found for '{query}' in {location_bias}."

    lines = []
    for p in places:
        name = p.get("displayName", {}).get("text", "Unknown")
        address = p.get("formattedAddress", "")
        rating = p.get("rating")
        count = p.get("userRatingCount")
        rating_str = f" - {rating}/5 ({count} reviews)" if rating else ""
        maps_url = p.get("googleMapsUri", "")
        lines.append(f"- {name}{rating_str}\n  {address}\n  {maps_url}")

    return "\n".join(lines)


# Tavily's LangChain tool is already a proper BaseTool with async support -
# no wrapper needed. Reads TAVILY_API_KEY from the environment.
web_search = TavilySearch(
    name="web_search",
    max_results=5,
    description=(
        "Search the web for up-to-date information not covered by the other "
        "tools - e.g. event listings, opening hours, reviews, or anything "
        "outside the assistant's training data. Do NOT use this for: "
        "Singapore weather/rain/temperature/forecast (use check_weather "
        "instead), PSI/haze/smoke/air quality (use check_psi instead), or "
        "UV/UV index/sun exposure (use check_uv_index instead) - those "
        "dedicated tools give live official NEA readings that this tool "
        "cannot beat."
    ),
)

ACTION_TOOLS = [check_weather, check_psi, check_uv_index, find_places, web_search]
