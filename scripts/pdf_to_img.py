import fitz  # PyMuPDF
import io
import yt.yson as yson
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gc
import time
import sys
import yt.wrapper as yt
from typing import Any, Iterable
from yt.wrapper.schema import RowIterator, Variant
from yt.wrapper import OutputRow


def pdf_bytes_to_images(pdf_bytes):
    """
    Convert PDF pages to PIL Images
    
    Args:
        pdf_bytes (bytes): PDF content as bytes
        
    Returns:
        list: List of PIL Image objects for each page
    """
    doc = fitz.open("pdf", pdf_bytes)
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    doc.close()
    return images


### --- Data Class for Bounding Box Records --- ###
@yt.yt_dataclass
class PdfImage:
    id: str | None
    page_index: int | None = None
    part_index: int | None = None
    content: bytes | None = None
    error: str | None = None

@yt.yt_dataclass
class PdfRecord:
    id: str | None
    part_index: int | None
    content: bytes | None

class PdfImageExtractor(yt.TypedJob):
    def __init__(self):
        self.IMAGE_PDF_TABLE_INDEX = 0

        self.max_pages = 1000


    def __call__(self, records: RowIterator[PdfRecord]) -> Iterable[OutputRow[Variant[PdfImage]]]:
        from pdf2image import convert_from_bytes

        ### --- Step 1: Reconstruct PDF from Rows --- ###
        id_ = None
        parts = []
        try:
            for record in records:
                id_ = record.id
                parts.append(record.content)
            pdf = b''.join(parts)
        except Exception as e:
            yield OutputRow(
                PdfImage(id=id_, error=f"Error reconstructing PDF: {str(e)}"),
                table_index=self.IMAGE_PDF_TABLE_INDEX,
            )
            return

        rows = []
        ### --- Step 2: Convert PDF to Images --- ###
        try:
            current_pdf_imgs = pdf_bytes_to_images(pdf)  # Convert PDF to images
            
            # Check for page limit
            if len(current_pdf_imgs) > self.max_pages:
                yield OutputRow(
                    PdfImage(id=id_, error=f"PDF exceeds maximum page limit of {self.max_pages}"),
                    table_index=self.IMAGE_PDF_TABLE_INDEX,
                )
                return

            ### --- Step 2.1: Process Each Page's Image Data --- ###
            for page, img in enumerate(current_pdf_imgs):
                with io.BytesIO() as img_byte_arr:
                    img.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()

                ptr = 0
                part_index = 0
                PART_SIZE = 1024 * 1024 * 100

                while ptr < len(img_bytes):
                    part_end = min(len(img_bytes), ptr + PART_SIZE)
                    part = img_bytes[ptr:part_end]
                    rows.append(OutputRow(
                        PdfImage(id=id_, page_index=page, part_index=part_index, content=part),
                        table_index=self.IMAGE_PDF_TABLE_INDEX,
                    ))
                    part_index += 1
                    ptr = part_end

        except Exception as e:
            yield OutputRow(
                PdfImage(id=id_, error=f"Error converting PDF to image: {str(e)}"),
                table_index=self.IMAGE_PDF_TABLE_INDEX,
            )
            return

        for row in rows:
            yield row


ytc = yt.YtClient(config=yt.default_config.get_config_from_env())
ytc.config["pickling"]["ignore_system_modules"] = True


input_path = ""
output_path = ""
ytc.create("table", output_path, attributes={"schema": [
    {"name": "id", "type": "string", "sort_order": "ascending"},
    {"name": "page_index", "type": "int64", "sort_order": "ascending"},
    {"name": "part_index", "type": "int64", "sort_order": "ascending"},
    {"name": "content", "type": "string"},  # Storing image part as string (binary data)
    {"name": "error", "type": "string"},  # Include error column for error handling
]}, ignore_existing=True)

ytc.run_reduce(
    PdfImageExtractor(),
    source_table=input_path,
    destination_table=output_path,
    reduce_by=["id"],
    sort_by=["id", "part_index"],
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
        # Add pool_trees parameter for GPU nodes
        "pool": "",
        #"pool_trees": ["default_gpu_h100"],
        "max_failed_job_count": 1,
        #"stderr_table_path": "//tmp/gritukan_stderr",
        #"job_count": 1,
    },
    sync=False,
)