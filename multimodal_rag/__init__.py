"""
Advanced Multimodal RAG System

A production-ready multimodal RAG system with advanced retrieval, 
reranking, and local LLM support.
"""

__version__ = "0.1.0"
__author__ = "Wzinh"
__email__ = "vinhquang2610345@gmail.com"

# Core imports
from .core.document_processor import MultimodalDocumentProcessor
from .core.embedder import MultimodalEmbedder  
from .core.vector_store import MultimodalVectorStore, VectorStoreConfig
from .core.retriever import HybridRetriever

# Advanced features
from .advanced.reranker import AdvancedReranker
from .advanced.record_manager import RecordManager
from .advanced.web_scraper import WebScraper

# Main system
from .system import MultimodalRAGSystem, AdvancedMultimodalRAGSystem

# Configuration
from .utils.config import RAGConfig, AdvancedRAGConfig

# Models
from .models.document import DocumentElement, ProcessedDocument
from .models.search import SearchResult, RerankedResult  
from .models.response import RAGResponse

# Integrations
from .integrations.ollama_client import OllamaClient, OllamaMultimodalRAGSystem

# Utilities
from .utils.logger import setup_logger

__all__ = [
    # Core components
    "MultimodalDocumentProcessor",
    "MultimodalEmbedder", 
    "MultimodalVectorStore",
    "VectorStoreConfig",
    "HybridRetriever",
    
    # Advanced features
    "AdvancedReranker",
    "RecordManager", 
    "WebScraper",
    
    # Main systems
    "MultimodalRAGSystem",
    "AdvancedMultimodalRAGSystem",
    
    # Configuration
    "RAGConfig",
    "AdvancedRAGConfig",
    
    # Models
    "DocumentElement",
    "ProcessedDocument",
    "SearchResult", 
    "RerankedResult",
    "RAGResponse",
    
    # Integrations
    "OllamaClient",
    "OllamaMultimodalRAGSystem",
    
    # Utilities
    "setup_logger",
]

# Package metadata
PACKAGE_INFO = {
    "name": "advanced-multimodal-rag",
    "version": __version__,
    "description": "Advanced Multimodal RAG System with Hybrid Search and Reranking",
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/vinhisreal/multimodal-rag#",
    "license": "MIT",
}
