import requests
from typing import Dict, Any, Optional


def get_each_model_details(model_id: str, api_key: Optional[str] = None) -> list:
    """
    Fetches the endpoints/routing data (and thus providers) for a specific model from OpenRouter.
    """
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json().get("data", {})
        return data.get("endpoints", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching endpoints for {model_id}: {e}")
        return []
