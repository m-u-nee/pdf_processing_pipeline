# Pleias Processing Pipeline
This repository showcases an implementation of a text extraction pipeline built with Pleias and deployed on Nebius AI. The pipeline is specifically designed to efficiently process and extract text content from large-scale PDF document collections.
## Overview
The pipeline leverages Pleias's advanced processing capabilities combined with Nebius AI's cloud infrastructure to handle high-volume PDF processing tasks. This solution is particularly suited for organizations needing to extract and analyze text from substantial document archives efficiently and accurately.
Key features:
- Large-scale PDF processing capabilities
- Optimized for deployment on Nebius AI infrastructure
- Built using Pleias processing framework
- Focused on accurate text extraction and processing
- Scripts are map-reduce ready for parallel processing
## Pipeline Walkthrough
### scripts/pdf_to_img.py
The first part of the pipeline converts PDFs (previously downloaded and stored in YT Saurus tables) into images, as the extraction process utilizes an image-based model. Since individual PDFs may be split across multiple table rows due to size constraints, this script runs as a reduce operation over unique PDF IDs.
The code utilizes PyMuPDF and Pillow to convert PDF bytes into images. Each PDF is processed page by page, with the resulting images being split into manageable chunks (100MB) and stored in YT tables. 

### scripts/img_to_box.py
The second part of the pipeline uses a YOLO model to detect and extract bounding boxes from the previously generated images. The script processes images in batches (128 or 256 depending on GPU type) for optimal performance, using CUDA when available. It implements a map-reduce operation that groups images by PDF ID and page index, running on GPU-enabled nodes (in our case H100s or L40s). The extractor outputs structured data containing bounding box coordinates, confidence scores, and class labels for each detected element in the PDF pages.

### scripts/box_to_text.py
The final part of the pipeline extracts and structures text from PDFs using the detected bounding boxes. It employs PyMuPDF to extract text within the identified regions and organizes content hierarchically based on text class types (Regular, Special, Margin, or Blank). The script intelligently manages text sections, splits large text chunks while preserving sentence integrity, and generates unique hashes for each text entry. The output is structured as JSON records containing section information, text content, page numbers, and metadata. The process runs as a reduce operation on CPU nodes, combining PDF content with bounding box data to produce the final structured text output.