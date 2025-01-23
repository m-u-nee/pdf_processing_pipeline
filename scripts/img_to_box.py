import fitz  # PyMuPDF
import io
import yt.yson as yson
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gc
import time
from IPython.display import display  # For displaying images in Jupyter
import sys
import yt.wrapper as yt
from typing import Any, Iterable
from yt.wrapper.schema import RowIterator, Variant
from yt.wrapper import OutputRow


class ImageProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if self.model is None:
            # NB: ultralytics logs to stdout during import and it breaks
            # reduce job since the output is parsed as a reducer output.
            # We overwrite stdout to fix it.
            import sys
            oo = sys.stdout
            sys.stdout = open("output.txt", "w")
            from ultralytics import YOLO
            sys.stdout = oo

            yolo = YOLO(self.model_path)
            # Specify device based on availability
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            yolo.to(device)
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            #self.model = torch.compile(yolo)
            self.model = yolo
            print(f"Model loaded on device: {yolo.device}", file=sys.stderr)

    def process_batch(self, image_batch):
        self.load_model()  # Ensure model is loaded
        
        # Process the entire batch at once
        results = self.model(image_batch, conf=0.12)
        
        batch_results = []
        
        for idx, (image, result) in enumerate(zip(image_batch, results)):
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().tolist()
            names = result.names
            
            data = []
            for box, class_id, confidence in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = box
                class_id = int(class_id)
                class_name = names[class_id]
                data.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': confidence,
                    'class_id': class_id, 'class_name': class_name
                })
            
            # Append the results for the current image
            batch_results.append({
                'image_index': idx,  # Using index instead of path
                'detections': data
            })

            # Clear memory for this iteration
            del boxes, class_ids, confidences, data
        
        # Clear batch results
        del results
        gc.collect()

        return batch_results

    def process_images(self, image_list):
        """
        Process a list of PIL Images and return the results.

        Args:
            image_list (list): List of PIL Image objects.

        Returns:
            list: List of dictionaries containing detection results.
        """
        if not image_list:
            return []

        all_results = []
        start_time = time.time()
        batch_size = 10  # Adjust as necessary for CPU/GPU processing

        # Process images in batches
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i:i + batch_size]
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)

            # Clear memory after each batch
            gc.collect()

        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.2f} seconds.", file=sys.stderr)
        return all_results





### --- Data Class for Bounding Box Records --- ###
@yt.yt_dataclass
class PdfImage:
    id: str | None
    page_index: int | None = None
    part_index: int | None = None
    content: bytes | None = None
    error: str | None = None

@yt.yt_dataclass
class BoundingBox:
    id: str | None
    label: str | None = None
    x_min: float | None = None
    y_min: float | None = None
    x_max: float | None = None
    y_max: float | None = None
    page_index: int | None = None
    confidence: float | None = None  # Added confidence field
    detection_index: int | None = None  # Added to track multiple detections per image
    error: str | None = None


class ImageBoundingBoxesExtractor(yt.TypedJob):
    def start(self):
        ### --- Initialization of the Extractor --- ###
        # from image_processing import pdf_bytes_to_images, ImageProcessor
        self.model_path = "/app/model/YoloV8-Detect.pt"
        import torch
        device_name = torch.cuda.get_device_name()
        if device_name == "NVIDIA L40S":
            self.batch_size = 128
        else:
            self.batch_size = 256
        # (img, page_index, id)
        self.batch = []
        self.image_processor = ImageProcessor(self.model_path)
        self.image_processor.load_model()

    ### --- Callable Method to Process PDF Rows --- ###
    def __call__(self, records: RowIterator[PdfImage]) -> Iterable[BoundingBox]:
        try:
            import io
            parts = []
            id_ = None
            page_index = None
            for record in records:
                if record.error:
                    return
                import sys
                id_ = record.id
                page_index = record.page_index
                parts.append(record.content)
            img = b''.join(parts)
            img = Image.open(io.BytesIO(img))
            img = img.convert("RGB")
        except Exception as e:
            print(str(e), file=sys.stderr)
            # We cannot write an error here due to potential sort error order violation.
            # yield BoundingBox(
            #     id=id_,
            #     page_index=page_index,
            #     error=f"Error extracting bounding boxes in process_batch: {str(e)}",
            # )
            return

        self.batch.append((img, page_index, id_))
        if len(self.batch) >= self.batch_size:
            yield from self.process_batch()
            self.batch = []

    def finish(self):
        yield from self.process_batch()

    ### --- Helper Method to Process a Batch of Images --- ###
    def process_batch(self):
        if len(self.batch) == 0:
            return

        rows = []
        try:
            # Extract the first batch_size images for processing
            batch_imgs = [img_tuple[0] for img_tuple in self.batch]
            import time
            start = time.time()
            results = self.image_processor.process_batch(batch_imgs)
            end = time.time()
            import sys
            print(end-start, file=sys.stderr)

            ### --- Step 3.1: Process Each Image's Detections --- ###
            for img_idx, result in enumerate(results):
                # Get the corresponding page index and PDF ID for this image
                _, corresponding_page, corresponding_id = self.batch[img_idx]

                # Process each detection in the image
                if 'detections' in result:
                    for det_idx, detection in enumerate(result['detections']):
                        rows.append(BoundingBox(
                            id=corresponding_id,
                            page_index=corresponding_page,  # Ensure page_index is listed second
                            detection_index=det_idx,  # Ensure detection_index follows
                            label=detection.get('class_name', ''),
                            x_min=float(detection['x1']),  # Convert numpy float32 to Python float
                            y_min=float(detection['y1']),
                            x_max=float(detection['x2']),
                            y_max=float(detection['y2']),
                            confidence=float(detection['confidence'])
                        ))

        except Exception as e:
            # Use the PDF ID from the first image in the batch for error reporting
            error_id = self.batch[0][2] if self.batch else "unknown"
            import sys
            print(str(e), file=sys.stderr)
            yield BoundingBox(
                id=error_id,
                error=f"Error extracting bounding boxes in process_batch: {str(e)}",
            )
            return
        for row in rows:
            yield row

ytc = yt.YtClient(config=yt.default_config.get_config_from_env())
ytc.config["pickling"]["ignore_system_modules"] = True

input_path = ""
output_path = ""

ytc.run_reduce(
    ImageBoundingBoxesExtractor(),
    source_table=input_path,
    destination_table=output_path,
    reduce_by=["id", "page_index"],
    sort_by=["id", "page_index", "part_index"],
    spec={
        "reducer": {
            "docker_image": "",
            # We work with large tars in-memory, so let's increase the default memory limit of 512MB.
            "memory_limit": 16 * 1024**3,
            # Add GPU limit as requested
            "gpu_limit": 1,
        },
        "job_io": {
            "table_writer": {
                # By default, the maximum possible size of a table cell is 16MB.
                # Let's increase it to the maximum possible value of 128MB.
                "max_row_weight": 128 * 1024**2,
            },
        },
        # Add pool_trees parameter for GPU nodes
        "pool_trees": ["default_gpu_h100", "default_gpu_l40s"],
        "max_failed_job_count": 1,
        # "stderr_table_path": "//tmp/gritukan_stderr",
        # "job_count": 1,
        "job_count": 1024,
    },
)