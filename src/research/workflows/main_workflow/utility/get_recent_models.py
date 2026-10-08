import requests
import time
from typing import Dict, Any, Optional


def get_recent_models(api_key: Optional[str] = None) -> list:
    """
    Fetches the list of all available models from the OpenRouter API created in the last 6 months.
    """
    url = "https://openrouter.ai/api/v1/models"
    headers = {}
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        models = data.get("data", [])
        
        # Calculate timestamp for 6 months ago (approx 180 days)
        six_months_ago = time.time() - (180 * 24 * 60 * 60)
        
        keys_to_extract = [
            "id", "canonical_slug", "name", "description", 
            "architecture", "per_request_limits", "links", "reasoning"
        ]
        
        recent_models = []
        for model in models:
            created_at = model.get("created", 0)
            if created_at >= six_months_ago:
                filtered_model = {k: model.get(k) for k in keys_to_extract if k in model}
                recent_models.append(filtered_model)
                
        return recent_models
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models from OpenRouter: {e}")
        return []
