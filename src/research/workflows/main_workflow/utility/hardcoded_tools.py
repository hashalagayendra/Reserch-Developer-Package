from typing import List, Dict, Any

# Array containing actual working AI tool data matching the AITool table structure.
# This data can be used to populate the ai_tools database table or load directly into the workflow.
# The table structure expects: tool_title, description, argument, api_url, http_method, headers.
ai_tools_data: List[Dict[str, Any]] = [
    {
        "tool_title": "geocodeLocation",
        "description": "Geocoding API to convert location names (e.g., 'Sri Lanka', 'Tokyo', 'New York') to latitude and longitude coordinates. Uses the free Open-Meteo Geocoding API which requires no API key.",
        "argument": {
            "name": {
                "type": "string",
                "description": "The location name to geocode, e.g., 'Sri Lanka', 'New York', 'Tokyo'",
                "required": True,
                "in": "query"
            },
            "count": {
                "type": "integer",
                "description": "Number of results to return (default: 1)",
                "required": False,
                "default": 1,
                "in": "query"
            },
            "language": {
                "type": "string",
                "description": "Language code for the response (default: 'en')",
                "required": False,
                "default": "en",
                "in": "query"
            },
            "format": {
                "type": "string",
                "description": "Response format (default: 'json')",
                "required": False,
                "default": "json",
                "in": "query"
            }
        },
        "api_url": "https://geocoding-api.open-meteo.com/v1/search",
        "http_method": "GET",
        "headers": {
            "Accept": "application/json"
        }
    },
    {
        "tool_title": "openMeteoWeather",
        "description": "Fetches current weather, hourly, and daily forecasts for a given latitude and longitude using Open-Meteo API. No API key required.",
        "argument": {
            "latitude": {
                "type": "number",
                "description": "Latitude of the location (e.g., 52.52)",
                "required": True,
                "in": "query"
            },
            "longitude": {
                "type": "number",
                "description": "Longitude of the location (e.g., 13.41)",
                "required": True,
                "in": "query"
            },
            "current_weather": {
                "type": "boolean",
                "description": "Set to true to include current weather in the response",
                "required": False,
                "default": True,
                "in": "query"
            },
            "hourly": {
                "type": "string",
                "description": "Comma-separated list of hourly variables (e.g., 'temperature_2m,relative_humidity_2m,wind_speed_10m')",
                "required": False,
                "in": "query"
            },
            "daily": {
                "type": "string",
                "description": "Comma-separated list of daily variables (e.g., 'temperature_2m_max,temperature_2m_min,precipitation_sum')",
                "required": False,
                "in": "query"
            },
            "timezone": {
                "type": "string",
                "description": "Timezone for the response (e.g., 'auto', 'America/New_York', 'Europe/Berlin')",
                "required": False,
                "default": "auto",
                "in": "query"
            },
            "forecast_days": {
                "type": "integer",
                "description": "Number of days to forecast (1-16). Default is 7.",
                "required": False,
                "default": 7,
                "in": "query"
            }
        },
        "api_url": "https://api.open-meteo.com/v1/forecast",
        "http_method": "GET",
        "headers": {
            "Accept": "application/json"
        }
    },
    {
        "tool_title": "getBitcoinPrice",
        "description": "Fetch current Bitcoin and cryptocurrency prices and details from CoinGecko free public API (no key required).",
        "argument": {
            "ids": {
                "type": "string",
                "description": "Comma-separated cryptocurrency ids (e.g., 'bitcoin', 'ethereum')",
                "required": False,
                "default": "bitcoin",
                "in": "query"
            },
            "vs_currencies": {
                "type": "string",
                "description": "The fiat currency to display price in, e.g., 'usd', 'eur', 'gbp', 'jpy'",
                "required": True,
                "default": "usd",
                "in": "query"
            }
        },
        "api_url": "https://api.coingecko.com/api/v3/simple/price",
        "http_method": "GET",
        "headers": {
            "Accept": "application/json"
        }
    }
]
