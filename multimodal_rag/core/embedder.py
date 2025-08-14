import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import hashlib

from multimodal_rag import DocumentElement, ProcessedDocument

@dataclass
class EmbeddingResult:
    """Container for embedding results"""
    element_id: str
    embedding: np.ndarray
    element_type: str
    metadata: Dict[str, Any]

class BaseEmbedder(ABC):
    """Abstract base class for embedders"""
    
    @abstractmethod
    def embed(self, content: Any) -> np.ndarray:
        pass

class TextEmbedder(BaseEmbedder):
    """Text embedding using sentence-transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._model = None
        self.logger = logging.getLogger(f'TextEmbedder-{model_name}')
    
    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self.logger.info(f"Loaded text embedding model: {self.model_name}")
            except ImportError:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        return self._model
    
    def embed(self, content: str) -> np.ndarray:
        """Embed text content"""
        if not isinstance(content, str):
            raise ValueError("TextEmbedder expects string input")
        
        # Preprocess text
        content = self._preprocess_text(content)
        
        # Generate embedding
        embedding = self.model.encode(content, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding"""
        # Basic cleaning
        text = text.strip()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text

class ImageEmbedder(BaseEmbedder):
    """Image embedding using CLIP"""
    
    def __init__(self, model_name: str = 'openai/clip-vit-large-patch14'):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self.logger = logging.getLogger(f'ImageEmbedder-CLIP')
    
    @property
    def model(self):
        if self._model is None:
            try:
                from transformers import CLIPVisionModel, CLIPProcessor
                self._model = CLIPVisionModel.from_pretrained(self.model_name)
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                self.logger.info(f"Loaded image embedding model: {self.model_name}")
            except ImportError:
                raise ImportError("transformers not installed. Run: pip install transformers")
        return self._model, self._processor
    
    def embed(self, content: Union[bytes, 'PIL.Image.Image']) -> np.ndarray:
        """Embed image content"""
        model, processor = self.model
        
        # Handle different input types
        if isinstance(content, bytes):
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(content))
        else:
            image = content
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate embedding
        with torch.no_grad():
            image_features = model(**inputs)
            embedding = image_features.pooler_output.squeeze().numpy()
        
        return embedding.astype(np.float32)

class TableEmbedder(BaseEmbedder):
    """Table embedding using specialized approach"""
    
    def __init__(self, approach: str = 'text_serialization'):
        self.approach = approach
        self.text_embedder = TextEmbedder()  # Fallback to text embedding
        self.logger = logging.getLogger('TableEmbedder')
    
    def embed(self, content: Dict[str, Any]) -> np.ndarray:
        """Embed table content"""
        if 'dataframe' in content:
            df = content['dataframe']
            return self._embed_dataframe(df)
        elif 'raw_data' in content:
            raw_data = content['raw_data']
            return self._embed_raw_table(raw_data)
        else:
            raise ValueError("Table content must contain 'dataframe' or 'raw_data'")
    
    def _embed_dataframe(self, df) -> np.ndarray:
        """Embed pandas DataFrame"""
        if self.approach == 'text_serialization':
            # Convert table to text representation
            table_text = self._dataframe_to_text(df)
            return self.text_embedder.embed(table_text)
        else:
            # Other approaches (e.g., specialized table embedders) can be added here
            raise NotImplementedError(f"Approach {self.approach} not implemented")
    
    def _embed_raw_table(self, raw_data: List[List[str]]) -> np.ndarray:
        """Embed raw table data"""
        # Convert to text representation
        table_text = self._raw_table_to_text(raw_data)
        return self.text_embedder.embed(table_text)
    
    def _dataframe_to_text(self, df) -> str:
        """Convert DataFrame to text representation"""
        text_parts = []
        
        # Add column headers
        text_parts.append("Table columns: " + ", ".join(df.columns.astype(str)))
        
        # Add sample rows (first 3 rows to avoid too much text)
        for idx, row in df.head(3).iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            text_parts.append(f"Row {idx + 1}: {row_text}")
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            text_parts.append("Numeric columns summary:")
            for col in numeric_cols:
                mean_val = df[col].mean()
                text_parts.append(f"{col} average: {mean_val:.2f}")
        
        return " | ".join(text_parts)
    
    def _raw_table_to_text(self, raw_data: List[List[str]]) -> str:
        """Convert raw table data to text"""
        if not raw_data:
            return "Empty table"
        
        text_parts = []
        
        # Headers
        if raw_data:
            text_parts.append("Headers: " + ", ".join(raw_data[0]))
        
        # Sample rows
        for i, row in enumerate(raw_data[1:4]):  # First 3 data rows
            row_text = " | ".join(row)
            text_parts.append(f"Row {i + 1}: {row_text}")
        
        return " | ".join(text_parts)

class MultimodalEmbedder:
    """
    Main class that coordinates embedding of different content types
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logger()
        
        # Initialize embedders
        self.text_embedder = TextEmbedder(self.config['text_model'])
        self.image_embedder = ImageEmbedder(self.config['image_model'])
        self.table_embedder = TableEmbedder(self.config['table_approach'])
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_file = Path(self.config.get('cache_file', 'embeddings_cache.json'))
        self._load_cache()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'text_model': 'all-MiniLM-L6-v2',
            'image_model': 'openai/clip-vit-large-patch14',
            'table_approach': 'text_serialization',
            'use_cache': True,
            'cache_file': 'embeddings_cache.json'
        }
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MultimodalEmbedder')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def embed_element(self, element: DocumentElement) -> EmbeddingResult:
        """Embed a single document element"""
        # Check cache first
        cache_key = self._get_cache_key(element)
        if self.config['use_cache'] and cache_key in self.embedding_cache:
            cached_result = self.embedding_cache[cache_key]
            return EmbeddingResult(
                element_id=element.element_id,
                embedding=np.array(cached_result['embedding']),
                element_type=element.element_type,
                metadata=cached_result['metadata']
            )
        
        # Generate new embedding
        try:
            if element.element_type == 'text':
                embedding = self.text_embedder.embed(element.content)
            elif element.element_type == 'image':
                # Handle image content
                if isinstance(element.content, dict) and 'image_data' in element.content:
                    embedding = self.image_embedder.embed(element.content['image_data'])
                else:
                    # Fallback to caption if available
                    caption = element.content.get('caption', 'Image') if isinstance(element.content, dict) else 'Image'
                    embedding = self.text_embedder.embed(caption)
            elif element.element_type == 'table':
                embedding = self.table_embedder.embed(element.content)
            else:
                # Fallback for unknown types
                self.logger.warning(f"Unknown element type: {element.element_type}, using text embedding")
                text_content = str(element.content)
                embedding = self.text_embedder.embed(text_content)
            
            result = EmbeddingResult(
                element_id=element.element_id,
                embedding=embedding,
                element_type=element.element_type,
                metadata={
                    'embedding_dim': embedding.shape[0],
                    'embedding_model': self._get_model_name(element.element_type),
                    'source_bbox': element.bbox,
                    'page_num': element.page_num
                }
            )
            
            # Cache the result
            if self.config['use_cache']:
                self.embedding_cache[cache_key] = {
                    'embedding': embedding.tolist(),
                    'metadata': result.metadata
                }
                self._save_cache()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to embed element {element.element_id}: {e}")
            raise
    
    def embed_document(self, processed_doc: ProcessedDocument) -> List[EmbeddingResult]:
        """Embed all elements in a processed document"""
        self.logger.info(f"Embedding document: {processed_doc.doc_id}")
        
        embeddings = []
        for element in processed_doc.elements:
            try:
                embedding_result = self.embed_element(element)
                embeddings.append(embedding_result)
            except Exception as e:
                self.logger.error(f"Failed to embed element {element.element_id}: {e}")
                continue
        
        self.logger.info(f"Successfully embedded {len(embeddings)}/{len(processed_doc.elements)} elements")
        return embeddings
    
    def _get_cache_key(self, element: DocumentElement) -> str:
        """Generate cache key for element"""
        content_str = str(element.content)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        return f"{element.element_type}_{content_hash[:16]}"
    
    def _get_model_name(self, element_type: str) -> str:
        """Get model name for element type"""
        if element_type == 'text':
            return self.config['text_model']
        elif element_type == 'image':
            return self.config['image_model']
        elif element_type == 'table':
            return f"table_{self.config['table_approach']}"
        else:
            return 'unknown'
    
    def _load_cache(self):
        """Load embedding cache from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.embedding_cache = json.load(f)
                self.logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.embedding_cache, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings"""
        return {
            'cache_size': len(self.embedding_cache),
            'text_model': self.config['text_model'],
            'image_model': self.config['image_model'],
            'table_approach': self.config['table_approach']
        }

# Demo and testing functions
def demo_embedding():
    """Demo the multimodal embedding system"""
    
    # Create sample elements for testing
    sample_elements = [
        DocumentElement(
            element_id="test_text_1",
            element_type="text",
            content="This is a sample text about artificial intelligence and machine learning technologies.",
            bbox=(100, 100, 500, 200),
            page_num=1,
            source_file="test.pdf",
            metadata={}
        ),
        DocumentElement(
            element_id="test_table_1",
            element_type="table",
            content={
                "raw_data": [
                    ["Product", "Price", "Sales"],
                    ["Product A", "100", "500"],
                    ["Product B", "200", "300"],
                    ["Product C", "150", "400"]
                ]
            },
            bbox=(100, 300, 500, 500),
            page_num=1,
            source_file="test.pdf",
            metadata={}
        )
    ]
    
    # Initialize embedder
    embedder = MultimodalEmbedder()
    
    print("Multimodal Embedding System Demo")
    print("================================")
    
    # Test embedding each element
    for element in sample_elements:
        try:
            result = embedder.embed_element(element)
            print(f"\nElement: {element.element_id}")
            print(f"Type: {result.element_type}")
            print(f"Embedding shape: {result.embedding.shape}")
            print(f"Embedding model: {result.metadata['embedding_model']}")
            print(f"First 5 values: {result.embedding[:5]}")
        except Exception as e:
            print(f"Error embedding {element.element_id}: {e}")
    
    # Show stats
    stats = embedder.get_embedding_stats()
    print(f"\nEmbedding Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    demo_embedding()
