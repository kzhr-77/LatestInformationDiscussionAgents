from langchain_community.tools.tavily_search import TavilySearchResults

def get_search_tool():
    # Requires TAVILY_API_KEY in env
    return TavilySearchResults(max_results=3)

