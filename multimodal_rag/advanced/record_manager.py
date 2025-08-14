from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from datetime import datetime

from multimodal_rag import SearchResult
@dataclass
class DocumentRecord:
    """Record for tracking document versions and avoiding duplicates"""
    doc_id: str
    source_path: str
    file_hash: str
    last_modified: datetime
    processing_time: float
    element_count: int
    metadata: Dict[str, Any]

@dataclass 
class RerankedResult:
    """Enhanced search result with reranking score"""
    search_result: SearchResult
    rerank_score: float
    original_rank: int
    final_rank: int
    rerank_method: str

class RecordManager:
    """
    Manages document records to prevent duplicates and track changes
    """
    
    def __init__(self, record_file: str = "document_records.json"):
        self.record_file = Path(record_file)
        self.records: Dict[str, DocumentRecord] = {}
        self.logger = logging.getLogger('RecordManager')
        self._load_records()
    
    def _load_records(self):
        """Load existing records from file"""
        if self.record_file.exists():
            try:
                with open(self.record_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, record_data in data.items():
                        # Convert timestamp back to datetime
                        record_data['last_modified'] = datetime.fromisoformat(record_data['last_modified'])
                        self.records[key] = DocumentRecord(**record_data)
                self.logger.info(f"Loaded {len(self.records)} document records")
            except Exception as e:
                self.logger.error(f"Failed to load records: {e}")
    
    def _save_records(self):
        """Save records to file"""
        try:
            serializable_records = {}
            for key, record in self.records.items():
                record_dict = asdict(record)
                # Convert datetime to string
                record_dict['last_modified'] = record.last_modified.isoformat()
                serializable_records[key] = record_dict
            
            with open(self.record_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save records: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate file hash for change detection"""
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def should_process_document(self, file_path: str) -> bool:
        """Check if document needs processing (new or changed)"""
        file_path = Path(file_path)
        doc_id = file_path.stem
        
        # If record doesn't exist, need to process
        if doc_id not in self.records:
            return True
        
        # Check if file has changed
        current_hash = self.get_file_hash(file_path)
        stored_hash = self.records[doc_id].file_hash
        
        return current_hash != stored_hash
    
    def add_record(self, doc_id: str, source_path: str, processing_time: float, 
                  element_count: int, metadata: Dict[str, Any] = None):
        """Add or update document record"""
        file_hash = self.get_file_hash(source_path)
        
        record = DocumentRecord(
            doc_id=doc_id,
            source_path=source_path,
            file_hash=file_hash,
            last_modified=datetime.now(),
            processing_time=processing_time,
            element_count=element_count,
            metadata=metadata or {}
        )
        
        self.records[doc_id] = record
        self._save_records()
        self.logger.info(f"Added record for {doc_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not self.records:
            return {"total_documents": 0}
        
        total_docs = len(self.records)
        total_elements = sum(r.element_count for r in self.records.values())
        avg_processing_time = sum(r.processing_time for r in self.records.values()) / total_docs
        
        return {
            "total_documents": total_docs,
            "total_elements": total_elements,
            "avg_processing_time": avg_processing_time,
            "last_processed": max(r.last_modified for r in self.records.values()).isoformat()
        }
