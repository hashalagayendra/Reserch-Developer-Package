import httpx
from typing import List, Dict, Any

class ToolDBClient:
    """
    Client for interacting with the Tool Database API.
    Used by the ToolDiscoveryWorkflow to search and store AI tools.
    """
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
    def search_tools(self, query_vector: List[float], limit: int = 2) -> List[Dict[str, Any]]:
        """
        Searches the database for tools using vector similarity.
        """
        endpoint = f"{self.base_url}/ai-tools/search"
        payload = {
            "query_vector": query_vector,
            "limit": limit
        }
        
        # Using synchronous httpx client as the current workflow node_explore_db is synchronous.
        # If the workflow becomes fully async, this can be updated to AsyncClient.
        with httpx.Client() as client:
            try:
                response = client.post(endpoint, json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                print(f"Error searching tools via API: {e}")
                return []
                
    def store_tool(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stores a new AI tool in the database.
        """
        endpoint = f"{self.base_url}/ai-tools/"
        
        with httpx.Client() as client:
            try:
                response = client.post(endpoint, json=tool_data, headers=self.headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                print(f"Error storing tool via API: {e}")
                return {}
