from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class DocumentElement:
    """Base class for document elements"""
    element_id: str
    element_type: str  # 'text', 'image', 'table', 'chart'
    content: Any
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    page_num: int
    source_file: str
    metadata: Dict[str, Any]

@dataclass
class ProcessedDocument:
    """Container for processed document with all elements"""
    doc_id: str
    source_path: str
    elements: List[DocumentElement]
    page_count: int
    metadata: Dict[str, Any]
