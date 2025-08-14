import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd

from multimodal_rag import DocumentElement, ProcessedDocument

class MultimodalDocumentProcessor:
    """
    Core document processor that handles PDF/DOCX files and extracts
    multimodal elements (text, images, tables, charts)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logger()
        
        # Initialize components (lazy loading)
        self._ocr_engine = None
        self._layout_model = None
        self._table_parser = None
        self._image_captioner = None
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'ocr_lang': ['en', 'vi'],  # Support Vietnamese
            'min_text_length': 20,
            'min_image_size': (50, 50),
            'table_detection_threshold': 0.8,
            'output_dir': './processed_docs'
        }
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MultimodalProcessor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    @property
    def ocr_engine(self):
        """Lazy load OCR engine"""
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',  # Can be extended to 'vi' 
                    show_log=False
                )
                self.logger.info("PaddleOCR initialized successfully")
            except ImportError:
                self.logger.warning("PaddleOCR not available, OCR disabled")
                self._ocr_engine = None
        return self._ocr_engine
    
    def process_document(self, file_path: str) -> ProcessedDocument:
        """
        Main processing pipeline for a document
        """
        self.logger.info(f"Processing document: {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and process accordingly
        if file_path.suffix.lower() == '.pdf':
            return self._process_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self._process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    def _process_pdf(self, file_path: Path) -> ProcessedDocument:
        """Process PDF document"""
        doc = fitz.open(file_path)
        elements = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_elements = self._extract_page_elements(page, page_num, str(file_path))
            elements.extend(page_elements)
        
        doc.close()
        
        return ProcessedDocument(
            doc_id=file_path.stem,
            source_path=str(file_path),
            elements=elements,
            page_count=len(doc),
            metadata={'file_size': file_path.stat().st_size}
        )
    
    def _extract_page_elements(self, page, page_num: int, source_file: str) -> List[DocumentElement]:
        """Extract all elements from a single page"""
        elements = []
        
        # 1. Extract text blocks
        text_elements = self._extract_text_blocks(page, page_num, source_file)
        elements.extend(text_elements)
        
        # 2. Extract images
        image_elements = self._extract_images(page, page_num, source_file)
        elements.extend(image_elements)
        
        # 3. Extract tables (simplified detection for now)
        table_elements = self._extract_tables(page, page_num, source_file)
        elements.extend(table_elements)
        
        return elements
    
    def _extract_text_blocks(self, page, page_num: int, source_file: str) -> List[DocumentElement]:
        """Extract text blocks from page"""
        elements = []
        text_dict = page.get_text("dict")
        
        for block_idx, block in enumerate(text_dict["blocks"]):
            if "lines" not in block:  # Skip image blocks
                continue
                
            block_text = ""
            bbox = block["bbox"]
            
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"] + " "
            
            block_text = block_text.strip()
            
            if len(block_text) >= self.config['min_text_length']:
                element = DocumentElement(
                    element_id=f"{source_file}_page{page_num}_text{block_idx}",
                    element_type="text",
                    content=block_text,
                    bbox=bbox,
                    page_num=page_num,
                    source_file=source_file,
                    metadata={"font_info": self._extract_font_info(block)}
                )
                elements.append(element)
        
        return elements
    
    def _extract_images(self, page, page_num: int, source_file: str) -> List[DocumentElement]:
        """Extract images from page"""
        elements = []
        image_list = page.get_images()
        
        for img_idx, img in enumerate(image_list):
            try:
                # Get image data
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    
                    # Get image bbox (approximate)
                    img_rect = page.get_image_bbox(img)
                    
                    # Generate caption if OCR is available
                    caption = self._generate_image_caption(img_data) if self.ocr_engine else "Image"
                    
                    element = DocumentElement(
                        element_id=f"{source_file}_page{page_num}_img{img_idx}",
                        element_type="image",
                        content={"image_data": img_data, "caption": caption},
                        bbox=img_rect,
                        page_num=page_num,
                        source_file=source_file,
                        metadata={"width": pix.width, "height": pix.height}
                    )
                    elements.append(element)
                
                pix = None
            except Exception as e:
                self.logger.warning(f"Failed to extract image {img_idx}: {e}")
        
        return elements
    
    def _extract_tables(self, page, page_num: int, source_file: str) -> List[DocumentElement]:
        """Extract tables from page (simplified implementation)"""
        elements = []
        
        # Simple table detection based on text alignment
        # This is a placeholder - in production, use TableTransformer or similar
        tables = page.find_tables()
        
        for table_idx, table in enumerate(tables):
            try:
                # Extract table data
                table_data = table.extract()
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                
                element = DocumentElement(
                    element_id=f"{source_file}_page{page_num}_table{table_idx}",
                    element_type="table",
                    content={"dataframe": df, "raw_data": table_data},
                    bbox=table.bbox,
                    page_num=page_num,
                    source_file=source_file,
                    metadata={"rows": len(table_data), "cols": len(table_data[0])}
                )
                elements.append(element)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract table {table_idx}: {e}")
        
        return elements
    
    def _process_docx(self, file_path: Path) -> ProcessedDocument:
        """Process DOCX document (placeholder)"""
        # Implement DOCX processing using python-docx
        # This would extract text, images, and tables from Word documents
        raise NotImplementedError("DOCX processing not yet implemented")
    
    def _extract_font_info(self, block: Dict) -> Dict[str, Any]:
        """Extract font information from text block"""
        font_info = {}
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    font_info["font"] = span.get("font", "")
                    font_info["size"] = span.get("size", 0)
                    font_info["flags"] = span.get("flags", 0)
                    break  # Just get first span info
                break
        return font_info
    
    def _generate_image_caption(self, image_data: bytes) -> str:
        """Generate caption for image using OCR"""
        if not self.ocr_engine:
            return "Image"
        
        try:
            # Convert bytes to numpy array for OCR
            import io
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            result = self.ocr_engine.ocr(image_array, cls=True)
            
            # Extract text from OCR result
            text_parts = []
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        text_parts.append(line[1][0])
            
            caption = " ".join(text_parts) if text_parts else "Image"
            return caption[:200]  # Limit caption length
            
        except Exception as e:
            self.logger.warning(f"Failed to generate image caption: {e}")
            return "Image"
    
    def save_processed_document(self, processed_doc: ProcessedDocument, output_dir: str = None) -> str:
        """Save processed document to JSON format"""
        output_dir = Path(output_dir or self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Prepare serializable data
        serializable_elements = []
        for element in processed_doc.elements:
            elem_data = {
                'element_id': element.element_id,
                'element_type': element.element_type,
                'bbox': element.bbox,
                'page_num': element.page_num,
                'source_file': element.source_file,
                'metadata': element.metadata
            }
            
            # Handle different content types
            if element.element_type == 'text':
                elem_data['content'] = element.content
            elif element.element_type == 'table':
                elem_data['content'] = {
                    'table_data': element.content['raw_data'],
                    'dataframe_shape': element.content['dataframe'].shape
                }
            elif element.element_type == 'image':
                elem_data['content'] = {
                    'caption': element.content.get('caption', 'Image'),
                    'has_image_data': 'image_data' in element.content
                }
            
            serializable_elements.append(elem_data)
        
        output_data = {
            'doc_id': processed_doc.doc_id,
            'source_path': processed_doc.source_path,
            'page_count': processed_doc.page_count,
            'metadata': processed_doc.metadata,
            'elements': serializable_elements
        }
        
        output_file = output_dir / f"{processed_doc.doc_id}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Processed document saved to: {output_file}")
        return str(output_file)
