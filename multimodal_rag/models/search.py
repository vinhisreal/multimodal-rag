from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class SearchResult:
    """Container for search results"""
    element_id: str
    content: Any
    element_type: str
    score: float
    metadata: Dict[str, Any]
    bbox: Tuple[float, float, float, float]
    page_num: int
    source_file: str