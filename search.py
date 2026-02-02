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

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "link": r.get("url", ""),
                "snippet": r.get("content", "")
            })

        return results

    except Exception as e:
        print("Tavily Web Search Error:", e)
        return []


def youtube_search(query: str, max_results: int = 4):
    try:
        yt_query = f"site:youtube.com {query}"

        response = tavily.search(
            query=yt_query,
            search_depth="basic",
            max_results=max_results
        )

        results = []
        for r in response.get("results", []):
            link = r.get("url", "")

            if "youtube.com/watch" in link or "youtu.be" in link:
                results.append({
                    "title": r.get("title", ""),
                    "link": link,
                    "snippet": r.get("content", "")
                })

        return results[:max_results]

    except Exception as e:
        print("Tavily YouTube Search Error:", e)
        return []
