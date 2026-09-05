import json
import logging
from typing import List, Dict, Any, Optional
from haystack import Document

logger = logging.getLogger(__name__)

class WebRetriever:
    """Retriever that fetches results from DuckDuckGo."""

    def __init__(self, max_results: int = 5, region: str = "wt-wt"):
        self.max_results = max_results
        self.region = region

    def retrieve(self, query: str, num_results: Optional[int] = None) -> List[Any]:
        """Search DuckDuckGo and return results as Haystack-compatible documents."""
        num = num_results or self.max_results
        
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                logger.error("DuckDuckGo search package not installed (pip install duckduckgo-search)")
                return []

        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(
                    query,
                    region=self.region,
                    max_results=num,
                ))
            
            results = []
            for i, r in enumerate(raw):
                results.append(Document(
                    content=r.get("body", ""),
                    id=f"web_{i}",
                    meta={
                        "source_filename": r.get("href", "web"),
                        "title": r.get("title", ""),
                        "source": r.get("href", "web"),
                        "url": r.get("href", ""),
                        "chunk_id": f"web_{i}"
                    },
                    score=0.9 - (i * 0.05)  # Slight decay for ranking
                ))
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
