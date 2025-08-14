import numpy as np
import chromadb
from chromadb.config import Settings
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from multimodal_rag import EmbeddingResult, MultimodalEmbedder
from multimodal_rag import DocumentElement, ProcessedDocument

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    persist_directory: str = "./chroma_db"
    collection_name: str = "multimodal_rag"
    distance_metric: str = "cosine"  # cosine, l2, ip
    max_results: int = 1000

class MultimodalVectorStore:
    """
    Vector database for storing and retrieving multimodal embeddings
    using ChromaDB as the backend
    """
    
    def __init__(self, config: VectorStoreConfig = None):
        self.config = config or VectorStoreConfig()
        self.logger = self._setup_logger()
        
        # Initialize ChromaDB client
        self.client = self._initialize_client()
        self.collection = self._get_or_create_collection()
        
        # Keep track of stored elements
        self.element_metadata = {}
        self._load_metadata()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MultimodalVectorStore')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _initialize_client(self) -> chromadb.Client:
        """Initialize ChromaDB client"""
        try:
            # Create persistent client
            client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            self.logger.info(f"Initialized ChromaDB client at {self.config.persist_directory}")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            self.logger.info(f"Retrieved existing collection: {self.config.collection_name}")
        except:
            # Create new collection
            collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            self.logger.info(f"Created new collection: {self.config.collection_name}")
        
        return collection
    
    def add_embeddings(self, embeddings: List[EmbeddingResult], 
                      document_elements: List[DocumentElement]) -> bool:
        """Add embeddings to vector store"""
        if len(embeddings) != len(document_elements):
            raise ValueError("Number of embeddings must match number of document elements")
        
        try:
            # Prepare data for ChromaDB
            ids = []
            embedding_vectors = []
            metadatas = []
            documents = []  # Text content for ChromaDB indexing
            
            for embedding, element in zip(embeddings, document_elements):
                ids.append(embedding.element_id)
                embedding_vectors.append(embedding.embedding.tolist())
                
                # Prepare metadata (ChromaDB requires JSON-serializable data)
                metadata = {
                    'element_type': element.element_type,
                    'page_num': element.page_num,
                    'source_file': element.source_file,
                    'bbox_x0': element.bbox[0],
                    'bbox_y0': element.bbox[1],
                    'bbox_x1': element.bbox[2],
                    'bbox_y1': element.bbox[3],
                    'embedding_dim': embedding.embedding.shape[0],
                    'has_content': True
                }
                metadatas.append(metadata)
                
                # Prepare document text (for ChromaDB's text search capabilities)
                if element.element_type == 'text':
                    doc_text = str(element.content)
                elif element.element_type == 'image':
                    # Use caption if available
                    if isinstance(element.content, dict):
                        doc_text = element.content.get('caption', f'Image from {element.source_file}')
                    else:
                        doc_text = f'Image from {element.source_file}'
                elif element.element_type == 'table':
                    # Use table summary
                    if isinstance(element.content, dict) and 'raw_data' in element.content:
                        headers = element.content['raw_data'][0] if element.content['raw_data'] else []
                        doc_text = f"Table with columns: {', '.join(headers)}"
                    else:
                        doc_text = f'Table from {element.source_file}'
                else:
                    doc_text = f'{element.element_type} from {element.source_file}'
                
                documents.append(doc_text)
                
                # Store full element data separately
                self.element_metadata[embedding.element_id] = {
                    'element': element,
                    'embedding_metadata': embedding.metadata
                }
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                embeddings=embedding_vectors,
                metadatas=metadatas,
                documents=documents
            )
            
            self.logger.info(f"Added {len(embeddings)} embeddings to vector store")
            self._save_metadata()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add embeddings: {e}")
            return False
    
    def search_by_embedding(self, query_embedding: np.ndarray, 
                           n_results: int = 5,
                           element_types: Optional[List[str]] = None,
                           source_files: Optional[List[str]] = None) -> List[SearchResult]:
        """Search by embedding vector"""
        try:
            # Prepare where clause for filtering
            where_clause = {}
            if element_types:
                where_clause['element_type'] = {'$in': element_types}
            if source_files:
                where_clause['source_file'] = {'$in': source_files}
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(n_results, self.config.max_results),
                where=where_clause if where_clause else None,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results['ids'] and results['ids'][0]:  # Check if results exist
                for i in range(len(results['ids'][0])):
                    element_id = results['ids'][0][i]
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    document_text = results['documents'][0][i]
                    
                    # Calculate similarity score (1 - cosine_distance)
                    score = 1.0 - distance if self.config.distance_metric == "cosine" else distance
                    
                    # Get full element data
                    element_data = self.element_metadata.get(element_id, {})
                    element = element_data.get('element')
                    
                    if element:
                        search_result = SearchResult(
                            element_id=element_id,
                            content=element.content,
                            element_type=metadata['element_type'],
                            score=score,
                            metadata={
                                **metadata,
                                'document_text': document_text,
                                'embedding_metadata': element_data.get('embedding_metadata', {})
                            },
                            bbox=(metadata['bbox_x0'], metadata['bbox_y0'], 
                                 metadata['bbox_x1'], metadata['bbox_y1']),
                            page_num=metadata['page_num'],
                            source_file=metadata['source_file']
                        )
                        search_results.append(search_result)
            
            self.logger.info(f"Found {len(search_results)} results for embedding search")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Failed to search by embedding: {e}")
            return []
    
    def search_by_text(self, query: str, n_results: int = 5,
                      element_types: Optional[List[str]] = None) -> List[SearchResult]:
        """Search by text query using ChromaDB's built-in text search"""
        try:
            # Prepare where clause
            where_clause = {}
            if element_types:
                where_clause['element_type'] = {'$in': element_types}
            
            # Use ChromaDB's text search
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.config.max_results),
                where=where_clause if where_clause else None,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Convert to SearchResult objects (similar to embedding search)
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    element_id = results['ids'][0][i]
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    document_text = results['documents'][0][i]
                    
                    score = 1.0 - distance if self.config.distance_metric == "cosine" else distance
                    
                    element_data = self.element_metadata.get(element_id, {})
                    element = element_data.get('element')
                    
                    if element:
                        search_result = SearchResult(
                            element_id=element_id,
                            content=element.content,
                            element_type=metadata['element_type'],
                            score=score,
                            metadata={
                                **metadata,
                                'document_text': document_text,
                                'query_match': query
                            },
                            bbox=(metadata['bbox_x0'], metadata['bbox_y0'], 
                                 metadata['bbox_x1'], metadata['bbox_y1']),
                            page_num=metadata['page_num'],
                            source_file=metadata['source_file']
                        )
                        search_results.append(search_result)
            
            self.logger.info(f"Found {len(search_results)} results for text query: '{query}'")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Failed to search by text: {e}")
            return []
    
    def get_by_source_file(self, source_file: str) -> List[SearchResult]:
        """Get all elements from a specific source file"""
        return self.search_by_text("", n_results=self.config.max_results, 
                                  element_types=None)  # This will be filtered by where clause
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get element type distribution
            type_distribution = {}
            for element_data in self.element_metadata.values():
                element_type = element_data['element'].element_type
                type_distribution[element_type] = type_distribution.get(element_type, 0) + 1
            
            return {
                'total_elements': count,
                'element_types': type_distribution,
                'collection_name': self.config.collection_name,
                'distance_metric': self.config.distance_metric,
                'persist_directory': self.config.persist_directory
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def delete_by_source_file(self, source_file: str) -> bool:
        """Delete all elements from a specific source file"""
        try:
            # Get all IDs for the source file
            results = self.collection.get(
                where={'source_file': source_file},
                include=['metadatas']
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                
                # Clean up metadata
                for element_id in results['ids']:
                    if element_id in self.element_metadata:
                        del self.element_metadata[element_id]
                
                self._save_metadata()
                self.logger.info(f"Deleted {len(results['ids'])} elements from {source_file}")
                return True
            else:
                self.logger.info(f"No elements found for source file: {source_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete elements from {source_file}: {e}")
            return False
    
    def _load_metadata(self):
        """Load element metadata from disk"""
        metadata_file = Path(self.config.persist_directory) / "element_metadata.pkl"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    self.element_metadata = pickle.load(f)
                self.logger.info(f"Loaded metadata for {len(self.element_metadata)} elements")
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {e}")
                self.element_metadata = {}
    
    def _save_metadata(self):
        """Save element metadata to disk"""
        metadata_file = Path(self.config.persist_directory) / "element_metadata.pkl"
        metadata_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.element_metadata, f)
        except Exception as e:
            self.logger.warning(f"Failed to save metadata: {e}")
    
    def reset_collection(self) -> bool:
        """Reset the entire collection (delete all data)"""
        try:
            self.client.delete_collection(self.config.collection_name)
            self.collection = self._get_or_create_collection()
            self.element_metadata = {}
            self._save_metadata()
            self.logger.info("Collection reset successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset collection: {e}")
            return False
