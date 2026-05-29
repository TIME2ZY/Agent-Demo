import asyncio
import html
import json
import re
from typing import Any, Awaitable, Callable
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from tools.registry import RegisteredTool


SearchFetcher = Callable[[str, int], Awaitable[list[dict[str, str]]]]


def validate_web_search_args(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("query must be a non-empty string")

    limit = arguments.get("limit", 5)
    if not isinstance(limit, int) or limit < 1 or limit > 10:
        raise ValueError("limit must be an integer between 1 and 10")

    return {
        "query": query,
        "limit": limit,
    }


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)


def _extract_results_from_html(payload: str, limit: int) -> list[dict[str, str]]:
    pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.DOTALL,
    )
    results: list[dict[str, str]] = []
    for match in pattern.finditer(payload):
        results.append(
            {
                "title": html.unescape(_strip_tags(match.group("title")).strip()),
                "url": html.unescape(match.group("url").strip()),
                "snippet": html.unescape(_strip_tags(match.group("snippet")).strip()),
            }
        )
        if len(results) >= limit:
            break
    return results


def _default_fetcher_sync(query: str, limit: int) -> list[dict[str, str]]:
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Agent-Demo/0.1)",
        },
    )
    with urlopen(request, timeout=15) as response:
        payload = response.read().decode("utf-8", errors="replace")

    results = _extract_results_from_html(payload, limit)
    if results:
        return results

    api_url = (
        "https://api.duckduckgo.com/"
        f"?q={quote_plus(query)}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
    )
    api_request = Request(
        api_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Agent-Demo/0.1)",
        },
    )
    with urlopen(api_request, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))

    fallback_results: list[dict[str, str]] = []
    abstract_text = str(data.get("AbstractText") or "").strip()
    abstract_url = str(data.get("AbstractURL") or "").strip()
    heading = str(data.get("Heading") or query).strip()
    if abstract_text and abstract_url:
        fallback_results.append(
            {
                "title": heading,
                "url": abstract_url,
                "snippet": abstract_text,
            }
        )
    return fallback_results[:limit]


async def default_fetcher(query: str, limit: int) -> list[dict[str, str]]:
    return await asyncio.to_thread(_default_fetcher_sync, query, limit)


def create_web_search_tool(fetcher: SearchFetcher | None = None) -> RegisteredTool:
    active_fetcher = fetcher or default_fetcher

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        results = await active_fetcher(arguments["query"], arguments["limit"])
        return {
            "ok": True,
            "query": arguments["query"],
            "results": results,
        }

    return RegisteredTool(
        name="web_search",
        description="Search the public web and return a short list of results with titles and snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        validator=validate_web_search_args,
        handler=handler,
    )
