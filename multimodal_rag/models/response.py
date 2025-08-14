from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from multimodal_rag import SearchResult
@dataclass
class RAGResponse:
    """Container for RAG response"""
    query: str
    answer: str
    source_elements: List[SearchResult]
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class RAGConfig:
    """Configuration for the complete RAG system"""
    # Document processing
    ocr_lang: List[str] = None
    min_text_length: int = 20
    
    # Embedding models
    text_model: str = 'all-MiniLM-L6-v2'
    image_model: str = 'openai/clip-vit-large-patch14'
    
    # Vector store
    persist_directory: str = "./multimodal_rag_db"
    collection_name: str = "multimodal_documents"
    
    # Retrieval
    default_k: int = 5
    max_context_length: int = 4000
    use_hybrid_search: bool = True
    
    # Generation
    llm_provider: str = "openai"  # openai, anthropic, local
    llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 512
    temperature: float = 0.7
    
    def __post_init__(self):
        if self.ocr_lang is None:
            self.ocr_lang = ['en', 'vi']
