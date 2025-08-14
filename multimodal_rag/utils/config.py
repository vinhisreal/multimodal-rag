import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class RAGConfig:
    """Base configuration for RAG system"""
    
    # Document processing
    ocr_lang: List[str] = field(default_factory=lambda: ['en', 'vi'])
    min_text_length: int = 20
    min_image_size: tuple = (50, 50)
    
    # Embedding models
    text_model: str = 'all-MiniLM-L6-v2'
    image_model: str = 'openai/clip-vit-large-patch14'
    table_approach: str = 'text_serialization'
    
    # Vector store
    persist_directory: str = "./multimodal_rag_db"
    collection_name: str = "multimodal_documents"
    distance_metric: str = "cosine"
    
    # Retrieval
    default_k: int = 5
    max_context_length: int = 4000
    use_hybrid_search: bool = True
    
    # Generation
    llm_provider: str = "ollama"  # ollama, openai, anthropic
    llm_model: str = "llama3.1"
    max_tokens: int = 512
    temperature: float = 0.7
    
    # Performance
    use_cache: bool = True
    cache_file: str = "embeddings_cache.json"
    batch_size: int = 32
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create config from environment variables"""
        config = cls()
        
        # Override with environment variables if present
        config.persist_directory = os.getenv("RAG_PERSIST_DIR", config.persist_directory)
        config.collection_name = os.getenv("RAG_COLLECTION_NAME", config.collection_name)
        config.text_model = os.getenv("RAG_TEXT_MODEL", config.text_model)
        config.llm_provider = os.getenv("RAG_LLM_PROVIDER", config.llm_provider)
        config.llm_model = os.getenv("RAG_LLM_MODEL", config.llm_model)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if self.default_k <= 0:
            issues.append("default_k must be positive")
        
        if self.max_context_length <= 100:
            issues.append("max_context_length too small")
            
        if self.temperature < 0 or self.temperature > 2:
            issues.append("temperature should be between 0 and 2")
            
        return issues

@dataclass  
class AdvancedRAGConfig(RAGConfig):
    """Extended configuration for advanced RAG features"""
    
    # Record management
    use_record_manager: bool = True
    record_file: str = "document_records.json"
    
    # Advanced reranking
    use_reranking: bool = True
    rerank_method: str = 'hybrid'  # 'cross_encoder', 'rrf', 'hybrid'
    rerank_top_k: int = 20
    cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    
    # Web scraping
    enable_web_scraping: bool = False
    scraping_urls: List[str] = field(default_factory=list)
    scraping_interval: int = 3600  # seconds
    
    # Contextual embeddings
    use_contextual_embeddings: bool = True
    context_window_size: int = 200  # characters
    
    # Performance monitoring
    enable_monitoring: bool = True
    log_queries: bool = True
    benchmark_mode: bool = False
    
    # Advanced processing
    enable_ocr: bool = True
    enable_table_extraction: bool = True
    enable_image_captioning: bool = True
    
    def __post_init__(self):
        """Post-initialization validation"""
        # Call parent's __post_init__ if it exists
        try:
            super().__post_init__()
        except AttributeError:
            pass
        
        # Adjust defaults for advanced features
        if self.use_reranking:
            self.max_context_length = min(self.max_context_length, 8000)
        
        # Ensure record file is in persist directory
        if not os.path.isabs(self.record_file):
            self.record_file = os.path.join(self.persist_directory, self.record_file)

@dataclass
class OllamaConfig:
    """Configuration for Ollama integration"""
    
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120
    stream: bool = False
    
    # Model-specific settings
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    
    @classmethod
    def from_env(cls) -> 'OllamaConfig':
        """Create Ollama config from environment variables"""
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "2048")),
        )