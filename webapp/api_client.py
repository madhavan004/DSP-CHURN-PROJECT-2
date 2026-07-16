"""
Thin wrapper around the FastAPI model service.

IMPORTANT: The webapp must NEVER import the model or connect to the
database directly — all data access goes through these HTTP calls.
"""
import os

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 15


class APIError(Exception):
    """Raised when the API is unreachable or returns an error response."""


def check_health() -> bool:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=TIMEOUT_SECONDS)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def predict(features: list[dict], source: str = "webapp") -> dict:
    """
    Call POST /predict with a list of feature dicts.
    Returns the parsed JSON response: {"predictions": [...], "count": N}
    Raises APIError on network issues or non-2xx responses.
    """
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"features": features, "source": source},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise APIError(f"Could not reach the model API at {API_URL}: {exc}") from exc

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise APIError(f"API returned {resp.status_code}: {detail}")

    return resp.json()


def get_past_predictions(
    start_date=None, end_date=None, source: str = "all", limit: int = 500
) -> dict:
    params = {"source": source, "limit": limit}
    if start_date is not None:
        params["start_date"] = start_date.isoformat()
    if end_date is not None:
        params["end_date"] = end_date.isoformat()

    try:
        resp = requests.get(
            f"{API_URL}/past-predictions", params=params, timeout=TIMEOUT_SECONDS
        )
    except requests.exceptions.RequestException as exc:
        raise APIError(f"Could not reach the model API at {API_URL}: {exc}") from exc

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise APIError(f"API returned {resp.status_code}: {detail}")

    return resp.json()
