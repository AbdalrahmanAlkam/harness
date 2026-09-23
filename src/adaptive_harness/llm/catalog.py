"""Small OpenRouter model catalog client for the interactive model picker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from urllib.request import Request, urlopen


CATALOG_URL = "https://openrouter.ai/api/v1/models?output_modalities=text"


@dataclass(frozen=True)
class CatalogModel:
    id: str
    name: str
    context_length: int = 0


def fetch_models(api_key: str | None = None, timeout: float = 8.0) -> list[CatalogModel]:
    """Fetch text models; keep credentials out of errors and logs."""
    headers = {"Accept": "application/json", "User-Agent": "adaptive-harness/2"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    with urlopen(Request(CATALOG_URL, headers=headers), timeout=timeout) as response:
        payload = json.load(response)
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise ValueError("OpenRouter returned an invalid model catalog")
    models = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        model_id = row["id"].strip()
        if not model_id:
            continue
        architecture = row.get("architecture") or {}
        if isinstance(architecture, dict) and "text" not in architecture.get("output_modalities", ["text"]):
            continue
        models.append(CatalogModel(model_id, str(row.get("name") or model_id),
                                   int(row.get("context_length") or 0)))
    return sorted(models, key=lambda item: item.id.casefold())
