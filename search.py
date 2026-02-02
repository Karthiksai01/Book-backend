import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def web_search(query: str, max_results: int = 5):
    try:
        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=max_results
        )

        urls = []

        for r in response.get("results", []):
            url = r.get("url", "")
            if url and "youtube.com" not in url and "youtu.be" not in url:
                urls.append(url)

        return urls[:max_results]

    except Exception as e:
        print("Web search error:", e)
        return []


def youtube_search(query: str, max_results: int = 4):
    try:
        yt_query = f"site:youtube.com {query}"

        response = tavily.search(
            query=yt_query,
            search_depth="basic",
            max_results=max_results
        )

        yt_urls = []

        for r in response.get("results", []):
            url = r.get("url", "")
            if "youtube.com/watch" in url or "youtu.be" in url:
                yt_urls.append(url)

        return yt_urls[:max_results]

    except Exception as e:
        print("YouTube search error:", e)
        return []
