"""Optional cited web search for research mode."""
from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from adaptive_harness.tools.base import Tool, ToolResult


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web and return titles, source URLs, and short snippets for research."
    parameters = {"type": "object", "properties": {"query": {"type": "string"},
                  "count": {"type": "integer", "default": 5}}, "required": ["query"]}

    def __init__(self, api_key: str | None = None, *, allow_public_metadata: bool = False):
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        self.allow_public_metadata = allow_public_metadata

    def execute(self, query: str, count: int = 5, **kwargs) -> ToolResult:
        if not self.api_key:
            if self.allow_public_metadata:
                return self._crossref_search(query, count)
            return ToolResult(success=False, output="", error="BRAVE_SEARCH_API_KEY is required for web search")
        if not isinstance(count, int) or isinstance(count, bool):
            return ToolResult(success=False, output="", error="Search count must be an integer")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(success=False, output="", error="Search query cannot be empty")
        url = "https://api.search.brave.com/res/v1/web/search?" + urlencode({"q": query, "count": max(1, min(count, 10))})
        request = Request(url, headers={"Accept": "application/json", "X-Subscription-Token": self.api_key})
        try:
            with urlopen(request, timeout=15) as response:
                payload = json.load(response)
            items = payload.get("web", {}).get("results", [])
            lines = [f"{item.get('title', 'Untitled')}\n{item.get('url', '')}\n{item.get('description', '')}"
                     for item in items]
            return ToolResult(success=True, output="\n\n".join(lines) or "No results found",
                              metadata={"result_count": len(lines), "source_urls": [item.get("url", "") for item in items]})
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Web search failed: {type(exc).__name__}: {exc}")

    @staticmethod
    def _crossref_search(query: str, count: int) -> ToolResult:
        """Search public scholarly metadata when a Brave key is not configured."""
        if not isinstance(query, str) or not query.strip():
            return ToolResult(success=False, output="", error="Search query cannot be empty")
        if not isinstance(count, int) or isinstance(count, bool):
            return ToolResult(success=False, output="", error="Search count must be an integer")
        url = "https://api.crossref.org/works?" + urlencode({
            "query.bibliographic": query, "rows": max(1, min(count, 10))})
        request = Request(url, headers={"Accept": "application/json",
                                        "User-Agent": "adaptive-harness/0.1 (research metadata search)"})
        try:
            with urlopen(request, timeout=15) as response:
                items = json.load(response).get("message", {}).get("items", [])
            lines = []
            urls = []
            for item in items:
                title = (item.get("title") or ["Untitled"])[0]
                doi = item.get("DOI", "")
                source = f"https://doi.org/{doi}" if doi else item.get("URL", "")
                if source:
                    urls.append(source)
                lines.append(f"{title}\n{source}\nDOI: {doi} (Crossref metadata only; inspect the paper before citing claims)")
            return ToolResult(success=True, output="\n\n".join(lines) or "No scholarly metadata found",
                              metadata={"result_count": len(lines), "source_urls": urls,
                                        "source": "Crossref public metadata"})
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Crossref search failed: {type(exc).__name__}: {exc}")
