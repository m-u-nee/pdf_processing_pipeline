
import fitz
import json
from typing import List, Dict, Union, BinaryIO
import io
from dataclasses import dataclass
import hashlib
import random
import re

@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    confidence: float
    detection_index: int
    page_index: int

class PDFTextExtractor:
    def __init__(self):
        self.current_section = None
        self.current_text = ""
    
    @staticmethod
    def generate_hash() -> str:
        """Generate a unique hash for each text entry."""
        random_string = str(random.randint(0, 1000000))
        return hashlib.md5(random_string.encode()).hexdigest()[:16]
    
    @staticmethod
    def classify_class_name(class_name: str) -> str:
        """Classify the text box type based on its class name."""
        if not class_name or class_name.isspace():
            return "Blank"
        elif "MarginText" in class_name:
            return "Margin"
        elif class_name == "MainZone-Head" or "Title" in class_name:
            if class_name not in ["RunningTitleZone", "PageTitleZone-Index"]:
                return "Special"
        return "Regular"
    
    @staticmethod
    def split_text(text: str, max_words: int = 400) -> List[str]:
        """Split text into chunks of maximum word count while preserving sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            if current_word_count + sentence_word_count > max_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

            # Prevent infinite loop by ensuring there's more than one sentence in current_chunk
            while current_word_count > max_words and len(current_chunk) > 1:
                chunks.append(' '.join(current_chunk[:-1]))
                last_sentence = current_chunk.pop()  # Pop instead of slicing
                current_chunk = [last_sentence]
                current_word_count = len(last_sentence.split())

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def create_entry(self, text: str, page_num: int, document_name: str, detection_index: int) -> Dict:
        """Create a JSON entry for the extracted text."""
        full_text = f"{self.current_section} {text}" if self.current_section else text
        return {
            "section": self.current_section,
            "text": full_text.strip(),
            "page": page_num,
            "document": document_name,
            "word_count": len(full_text.split()),
            "hash": self.generate_hash(),
            "detection_index": detection_index
        }

    def extract_text_from_page(self, page: fitz.Page, boxes: List[BoundingBox], 
                             page_num: int, document_name: str) -> List[Dict]:
        """Extract text from a single page using the provided bounding boxes."""
        json_data = []
        
        # Sort boxes by y_min coordinate for top-to-bottom reading
        boxes = sorted(boxes, key=lambda x: x.y_min)
        
        for box in boxes:
            rect = fitz.Rect(box.x_min, box.y_min, box.x_max, box.y_max)
            text = page.get_text("text", clip=rect).strip()
            category = self.classify_class_name(box.label)
            
            if category in ["Blank", "Margin"]:
                json_data.append({
                    "section": text,
                    "text": text,
                    "page": page_num,
                    "document": document_name,
                    "word_count": len(text.split()),
                    "hash": self.generate_hash(),
                    "detection_index": box.detection_index
                })
                continue

            elif category == "Special":
                if self.current_section is not None and self.current_text:
                    for chunk in self.split_text(self.current_text):
                        json_data.append(self.create_entry(chunk, page_num, document_name, box.detection_index))
                self.current_section = text
                self.current_text = ""

            else:  # Regular
                if self.current_section is None:
                    self.current_section = text
                
                self.current_text = f"{self.current_text} {text}" if self.current_text else text
                
                full_text = f"{self.current_section} {self.current_text}" if self.current_section else self.current_text
                if len(full_text.split()) > 400:
                    chunks = self.split_text(self.current_text)
                    for chunk in chunks[:-1]:
                        json_data.append(self.create_entry(chunk, page_num, document_name, box.detection_index))
                    self.current_text = chunks[-1]

        return json_data

    def process_pdf(self, pdf_data: Union[bytes, BinaryIO], bounding_boxes: List[Dict], 
                   document_name: str = "document") -> List[Dict]:
        """
        Process PDF data from memory and extract text using provided bounding boxes.
        
        Args:
            pdf_data: PDF content as bytes or BytesIO object
            bounding_boxes: List of bounding box dictionaries from YT
            document_name: Name of the document (optional)
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        # Reset instance variables
        self.current_section = None
        self.current_text = ""
        
        # Convert YT bounding boxes to our format and organize by page
        boxes_by_page = {}
        for bb in bounding_boxes:
            box = BoundingBox(
                x_min=bb.x_min,
                y_min=bb.y_min,
                x_max=bb.x_max,
                y_max=bb.y_max,
                label=bb.label,
                confidence=bb.confidence,
                detection_index=bb.detection_index,
                page_index=bb.page_index
            )
            
            if box.page_index not in boxes_by_page:
                boxes_by_page[box.page_index] = []
            boxes_by_page[box.page_index].append(box)
        
        all_json_data = []
        
        # Open PDF from memory
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        
        try:
            # Process each page that has bounding boxes
            for page_num in sorted(boxes_by_page.keys()):
                if 0 <= page_num < len(doc):
                    page = doc[page_num]
                    json_data = self.extract_text_from_page(
                        page, boxes_by_page[page_num], page_num + 1, document_name
                    )
                    all_json_data.extend(json_data)
            
            # Process any remaining text in the buffer
            if self.current_section is not None and self.current_text:
                for chunk in self.split_text(self.current_text):
                    all_json_data.append(self.create_entry(
                        chunk, max(boxes_by_page.keys()) + 1, document_name, -1  # -1 if detection index isn't applicable
                    ))
            
        finally:
            doc.close()
        
        return all_json_data


import yt.wrapper
import yt.wrapper.schema as schema
import typing
from yt.wrapper.schema import RowIterator, OutputRow, Variant
import yt.yson as yson


@yt.wrapper.yt_dataclass
class PDFRow:
    id: str | None  # Allow for optional id to match schema
    content: bytes | None  # Set as optional in case some PDFs lack content
    part_index: int | None


@yt.wrapper.yt_dataclass
class BoundingBoxRow:
    id: str | None  # Document ID should always be present
    page_index: int | None
    detection_index: int | None
    label: str | None
    x_min: float | None
    y_min: float | None
    x_max: float | None
    y_max: float | None
    confidence: float | None
    error: str | None  # Optional field for error handling


@yt.wrapper.yt_dataclass
class PDFBoundingBoxOutputRow:
    id: str | None
    page: int | None
    detection_index: int | None
    section: str | None
    text: str | None
    word_count: int | None
    hash: str | None
    error: str | None


class PDFBoundingBoxReducer(yt.wrapper.TypedJob):
    PDF_TABLE_INDEX = 0
    BB_TABLE_INDEX = 1

    def __init__(self):
        self.extractor = PDFTextExtractor()

    def process_pdf_with_boxes(self, pdf_bytes: bytes, bounding_boxes: list, target_id: str) -> list:
        try:
            return self.extractor.process_pdf(
                pdf_data=pdf_bytes,
                bounding_boxes=bounding_boxes,
                document_name=target_id
            )
        except Exception as e:
            return [{"id": target_id, "error": str(e)}]

    def __call__(self, input_row_iterator: RowIterator[Variant[PDFRow, BoundingBoxRow]]) \
            -> typing.Iterable[OutputRow[PDFBoundingBoxOutputRow]]:
        pdf_parts = {}
        bounding_boxes = []
        current_id = None
        
        # Collect all related rows
        for input_row, context in input_row_iterator.with_context():
            if context.get_table_index() == self.PDF_TABLE_INDEX:
                current_id = input_row.id
                if input_row.content is not None:
                    part_index = input_row.part_index if input_row.part_index is not None else 0
                    pdf_parts[part_index] = yson.get_bytes(input_row.content)
            elif context.get_table_index() == self.BB_TABLE_INDEX:
                bounding_boxes.append(input_row)
        
        if not pdf_parts or not bounding_boxes:
            return
            
        # Assemble PDF parts in order
        assembled_pdf_bytes = b''.join(pdf_parts[index] for index in sorted(pdf_parts))
        
        # Process PDF with bounding boxes
        results = self.process_pdf_with_boxes(assembled_pdf_bytes, bounding_boxes, current_id)
        
        # Yield results as instances of PDFBoundingBoxOutputRow
        for result in results:
            yield PDFBoundingBoxOutputRow(
                id=current_id,
                page=result.get("page", None),

                detection_index=result.get("detection_index", None),
                section=result.get("section", None),
                text = result.get("text", None),
                word_count=result.get("word_count", None),
                hash=result.get("hash", None),
                error=result.get("error", None)
            )


client = yt.wrapper.YtClient(proxy="charlie.yt.nebius.yt", config={"backend": "rpc"})

PDF_TABLE_PATH = ""
BB_TABLE_PATH = ""
OUTPUT_TABLE_PATH = ""



client.run_reduce(
        PDFBoundingBoxReducer(),
        source_table=[PDF_TABLE_PATH, BB_TABLE_PATH],
        destination_table=OUTPUT_TABLE_PATH,
        reduce_by=["id"],
        sync = False,
        spec={
            "reducer": {
                # We need to use the same docker image as for the Jupyter kernel.
                "docker_image": "",
                # We work with large tars in-memory, so let's increase the default memory limit of 512MB.
                "memory_limit": 4 * 1024**3,
                # Add GPU limit as requested
                #"gpu_limit": 1,
            },
            "job_io": {
                "table_writer": {
                    # By default, the maximum possible size of a table cell is 16MB.
                    # Let's increase it to the maximum possible value of 128MB.
                    "max_row_weight": 128 * 1024**2,
                },
            },
            "pool": "mattia_fifo",
            
            #"pool_trees": ["default_gpu_h100"],
            "max_failed_job_count": 1,
            #"stderr_table_path": "//tmp/gritukan_stderr",
            "job_count": 400,
        },
    )