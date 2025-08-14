import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .utils.config import RAGConfig, AdvancedRAGConfig
from .utils.logger import setup_logger
from .core.document_processor import MultimodalDocumentProcessor
from .core.embedder import MultimodalEmbedder
from .core.vector_store import MultimodalVectorStore, VectorStoreConfig
from .core.retriever import HybridRetriever
from .models.response import RAGResponse
from .models.search import SearchResult

class MultimodalRAGSystem:
    """
    Main RAG system combining all components
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.logger = setup_logger('MultimodalRAGSystem')
        
        # Initialize components
        self.logger.info("Initializing Multimodal RAG System...")
        
        # Document processor
        processor_config = {
            'ocr_lang': self.config.ocr_lang,
            'min_text_length': self.config.min_text_length
        }
        self.document_processor = MultimodalDocumentProcessor(processor_config)
        
        # Embedder
        embedder_config = {
            'text_model': self.config.text_model,
            'image_model': self.config.image_model,
            'table_approach': self.config.table_approach,
            'use_cache': self.config.use_cache,
            'cache_file': self.config.cache_file
        }
        self.embedder = MultimodalEmbedder(embedder_config)
        
        # Vector store
        vector_config = VectorStoreConfig(
            persist_directory=self.config.persist_directory,
            collection_name=self.config.collection_name,
            distance_metric=self.config.distance_metric
        )
        self.vector_store = MultimodalVectorStore(vector_config)
        
        # Retriever
        self.retriever = HybridRetriever(self.vector_store, self.embedder)
        
        self.logger.info("✅ Multimodal RAG System initialized successfully!")
    
    def ingest_document(self, file_path: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Ingest a document into the RAG system"""
        start_time = time.time()
        
        try:
            self.logger.info(f"📄 Ingesting document: {file_path}")
            
            # Process document
            processed_doc = self.document_processor.process_document(file_path)
            self.logger.info(f"   Extracted {len(processed_doc.elements)} elements")
            
            # Generate embeddings
            embeddings = self.embedder.embed_document(processed_doc)
            self.logger.info(f"   Generated {len(embeddings)} embeddings")
            
            # Store in vector database
            success = self.vector_store.add_embeddings(embeddings, processed_doc.elements)
            
            if success:
                processing_time = time.time() - start_time
                self.logger.info(f"✅ Document ingested successfully in {processing_time:.2f}s")
                return True
            else:
                self.logger.error("❌ Failed to store embeddings")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to ingest document: {e}")
            return False
    
    def query(self, query: str, 
              n_results: int = None,
              element_types: Optional[List[str]] = None) -> RAGResponse:
        """Query the RAG system"""
        start_time = time.time()
        n_results = n_results or self.config.default_k
        
        try:
            self.logger.info(f"🔍 Processing query: {query}")
            
            # Retrieve relevant elements
            if self.config.use_hybrid_search:
                search_results = self.retriever.hybrid_search(
                    query, n_results=n_results, element_types=element_types
                )
            else:
                search_results = self.vector_store.search_by_text(
                    query, n_results=n_results, element_types=element_types
                )
            
            # Generate answer (placeholder - implement with actual LLM)
            answer = self._generate_answer(query, search_results)
            
            processing_time = time.time() - start_time
            
            response = RAGResponse(
                query=query,
                answer=answer,
                source_elements=search_results,
                processing_time=processing_time,
                metadata={
                    'n_results_requested': n_results,
                    'n_results_found': len(search_results),
                    'element_types_found': list(set(r.element_type for r in search_results)),
                    'search_method': 'hybrid' if self.config.use_hybrid_search else 'text'
                }
            )
            
            self.logger.info(f"✅ Query processed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process query: {e}")
            return RAGResponse(
                query=query,
                answer=f"Error processing query: {str(e)}",
                source_elements=[],
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _generate_answer(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate answer from search results (placeholder)"""
        if not search_results:
            return "No relevant information found."
        
        # Placeholder implementation
        context_parts = []
        for i, result in enumerate(search_results[:3]):
            source_name = Path(result.source_file).name
            if result.element_type == 'text':
                snippet = str(result.content)[:200] + "..." if len(str(result.content)) > 200 else str(result.content)
                context_parts.append(f"{i+1}. From {source_name}: {snippet}")
            else:
                context_parts.append(f"{i+1}. From {source_name}: [{result.element_type} content]")
        
        return f"Based on the documents, here's what I found:\n\n" + "\n\n".join(context_parts)

class AdvancedMultimodalRAGSystem(MultimodalRAGSystem):
    """
    Advanced RAG system with additional features like reranking and record management
    """
    
    def __init__(self, config: AdvancedRAGConfig = None):
        self.advanced_config = config or AdvancedRAGConfig()
        super().__init__(self.advanced_config)
        
        # Initialize advanced components
        if self.advanced_config.use_record_manager:
            from .advanced.record_manager import RecordManager
            self.record_manager = RecordManager(self.advanced_config.record_file)
        
        if self.advanced_config.use_reranking:
            from .advanced.reranker import AdvancedReranker
            rerank_config = {
                'use_cross_encoder': self.advanced_config.rerank_method in ['cross_encoder', 'hybrid'],
                'cross_encoder_model': self.advanced_config.cross_encoder_model,
                'max_rerank_candidates': self.advanced_config.rerank_top_k
            }
            self.reranker = AdvancedReranker(rerank_config)
        
        self.logger.info("🚀 Advanced Multimodal RAG System initialized!")
    
    def query(self, query: str, 
              n_results: int = None,
              element_types: Optional[List[str]] = None,
              use_reranking: bool = None) -> RAGResponse:
        """Enhanced query with advanced reranking"""
        use_reranking = use_reranking if use_reranking is not None else self.advanced_config.use_reranking
        
        # Get base response
        response = super().query(query, n_results, element_types)
        
        # Apply reranking if enabled
        if use_reranking and hasattr(self, 'reranker') and response.source_elements:
            reranked_results = self.reranker.rerank_results(
                query, response.source_elements, method=self.advanced_config.rerank_method
            )
            
            # Update response with reranked results
            response.source_elements = [r.search_result for r in reranked_results[:n_results or self.config.default_k]]
            response.metadata['rerank_method'] = self.advanced_config.rerank_method
            response.metadata['reranked'] = True
        
        return response
