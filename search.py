from ddgs import DDGS

def web_search(query: str, max_results: int = 5):
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "link": r.get("href", ""),
                "snippet": r.get("body", "")
            })

    return results


def youtube_search(query: str, max_results: int = 4):
    results = []

    # ✅ Search YouTube results via normal web search query
    yt_query = f"site:youtube.com {query}"

    with DDGS() as ddgs:
        for r in ddgs.text(yt_query, max_results=max_results):
            link = r.get("href", "")
            if "youtube.com/watch" in link or "youtu.be" in link:
                results.append({
                    "title": r.get("title", ""),
                    "link": link,
                    "snippet": r.get("body", "")
                })

    return results[:max_results]
