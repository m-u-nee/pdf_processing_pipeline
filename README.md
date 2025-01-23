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