import requests
from typing import Dict, Any, Optional


def get_design_arena_benchmarks(api_key: Optional[str] = None) -> list:
    """
    Fetches the benchmarks from OpenRouter API, filtered for the Design Arena source.
    """
    url = "https://openrouter.ai/api/v1/benchmarks"
    params = {"source": "design-arena"}
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching benchmarks from OpenRouter: {e}")
        return []
