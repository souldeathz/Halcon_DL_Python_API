from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
import base64
import os
import time
from PIL import Image
import io
import json
from uuid import uuid4
from collections import OrderedDict
import halcon as ha
import tempfile
import traceback
import uuid

TEMP_FOLDER = os.path.join(os.path.dirname(__file__), "Temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

app = FastAPI()


# Load config
with open("config.json", "r") as f:
    config = json.load(f)

MODEL_BASE_PATH = config["MODEL_BASE_PATH"]

# TTL cache for mask_store
class ExpiringDict(OrderedDict):
    def __init__(self, max_age_seconds=300):
        self.max_age = max_age_seconds
        super().__init__()

    def __setitem__(self, key, value):
        super().__setitem__(key, (time.time(), value))
        self._cleanup()

    def __getitem__(self, key):
        ts, value = super().__getitem__(key)
        if time.time() - ts > self.max_age:
            raise KeyError
        return value

    def _cleanup(self):
        now = time.time()
        keys_to_delete = [k for k, (ts, _) in self.items() if now - ts > self.max_age]
        for k in keys_to_delete:
            del self[k]

mask_store = ExpiringDict()

# -------- Models --------
class DataInfoItem(BaseModel):
    Bbox_Class_ID: int
    Bbox_Class_Name: str
    Bbox_Confidence: float
    Bbox_Row1: float
    Bbox_Col1: float
    Bbox_Row2: float
    Bbox_Col2: float
    Mask_image_base64: str
    Mask_image_preview_url: str = None

class QueryResponse(BaseModel):
    Status_Code: str
    Status_Message: str
    DataInfo: List[DataInfoItem]
    Processing_Time: float
    Client_ID: str
    Timestamp: str


def create_dl_sample_batch(images: ha.HObject) -> list:
    # Count number of images in the tuple
    num_images = ha.count_obj(images)

    # Initialize empty list for DLSampleBatch
    dl_sample_batch = []

    for i in range(num_images):
        # Select each image from the tuple
        image = ha.select_obj(images, i + 1)

        # Create dictionary for DLSample
        dl_sample = ha.create_dict()

        # Set image into the dictionary under key 'image'
        ha.set_dict_object(image, dl_sample, 'image')

        # Add to batch
        dl_sample_batch.append(dl_sample)

    return dl_sample_batch

# -------- Shared processing logic --------
# Updated process_inference with HALCON image loading and DLSampleBatch creation
def process_inference(image_bytes: bytes, model: str, client_id: str, file_ext: str, include_preview_url: bool = True):
    start_time = time.time()

    try:
        Image.open(io.BytesIO(image_bytes)).verify()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    except Exception:
        return JSONResponse(
            status_code=200,
            content={
                "Status_Code": "01",
                "Status_Message": "Invalid image format or decode error",
                "Processing_Time": 0,
                "Client_ID": client_id,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "DataInfo": []
            }
        )

    model_file = os.path.join(MODEL_BASE_PATH, model + ".hdl")
    if not os.path.isfile(model_file):
        return JSONResponse(
            status_code=200,
            content={
                "Status_Code": "02",
                "Status_Message": f"Model '{model}' not found or missing .hdl file",
                "Processing_Time": 0,
                "Client_ID": client_id,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "DataInfo": []
            }
        )

    # ✅ ใช้ file_ext ที่รับมา
    temp_filename = f"input_{uuid4().hex[:8]}{file_ext}"
    temp_path = os.path.join(TEMP_FOLDER, temp_filename)
    with open(temp_path, "wb") as f:
        f.write(image_bytes)

    print(f"Processing image with model: {model}, temp file created at {temp_path}")

    try:
        ho_image = ha.read_image(temp_path)

        dl_sample_batch = create_dl_sample_batch(ho_image)
        # sample_batch = ha.create_dict()
        # ha.set_dict_object(ho_image, sample_batch, "image")
        # model_handle = ha.read_dl_model(model_file)
        # results = ha.apply_dl_model(model_handle, sample_batch)

    except Exception as e:
        error_message = f"HALCON Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        return JSONResponse(
            status_code=200,
            content={
                "Status_Code": "99",
                "Status_Message": error_message,
                "DataInfo": [],
                "Processing_Time": 0,
                "Client_ID": client_id,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )

    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as cleanup_error:
            print(f"[WARNING] Failed to delete temp file: {cleanup_error}")

    result_list = []
    for i in range(3):
        mask_id = f"mask_{uuid4().hex[:8]}"
        mask_store[mask_id] = image_base64

        item = {
            "Bbox_Class_ID": i,
            "Bbox_Class_Name": f"Class {i+1}",
            "Bbox_Confidence": 0.85,
            "Bbox_Row1": 100 + i * 10,
            "Bbox_Col1": 150 + i * 10,
            "Bbox_Row2": 200 + i * 10,
            "Bbox_Col2": 250 + i * 10,
            "Mask_image_base64": image_base64,
        }

        if include_preview_url:
            item["Mask_image_preview_url"] = f"http://127.0.0.1:8000/mask-preview/{mask_id}"

        result_list.append(item)

    processing_time_ms = (time.time() - start_time) * 1000

    return {
        "Status_Code": "OK",
        "Status_Message": "",
        "Processing_Time": round(processing_time_ms, 4),
        "Client_ID": client_id,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "DataInfo": result_list
    }
# -------- Endpoint 1: Upload File --------
@app.post("/query-upload", response_model=QueryResponse, summary="Upload image file and get result with preview URL")
async def query_upload(
    image_file: UploadFile = File(..., description="Image file (.jpg, .png, etc.)"),
    model: str = Form(..., description="Model name"),
    client_id: str = Form(..., description="Client ID")
):
    allowed_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    file_ext = os.path.splitext(image_file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return JSONResponse(
            status_code=200,
            content={
                "Status_Code": "03",
                "Status_Message": f"Unsupported file type '{file_ext}'",
                "Processing_Time": 0,
                "Client_ID": client_id,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "DataInfo": []
            }
        )

    image_bytes = await image_file.read()
    return process_inference(image_bytes, model, client_id, file_ext, include_preview_url=True)

# -------- Endpoint 2: Base64 Image --------
class Base64ImageRequest(BaseModel):
    image_base64: str
    model: str
    client_id: str

@app.post("/query-base64", response_model=QueryResponse, summary="Upload base64 image and get result (no preview URL)")
async def query_base64(request: Base64ImageRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception:
        return JSONResponse(
            status_code=200,
            content={
                "Status_Code": "01",
                "Status_Message": "Base64 decode error",
                "Processing_Time": 0,
                "Client_ID": request.client_id,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "DataInfo": []
            }
        )

    return process_inference(image_bytes, request.model, request.client_id, include_preview_url=False)

# -------- Preview mask --------
@app.get("/mask-preview/{mask_id}", response_class=HTMLResponse)
def preview_mask(mask_id: str):
    try:
        b64 = mask_store[mask_id]
    except KeyError:
        return HTMLResponse("<h3>Mask expired or not found</h3>", status_code=404)

    html = f'''
    <html><body>
        <h3>Mask Preview: {mask_id}</h3>
        <img src="data:image/png;base64,{b64}" style="max-width: 100%; border: 1px solid #ccc;" />
    </body></html>
    '''
    return HTMLResponse(content=html)

# -------- Model list --------
@app.get("/model-list", summary="List available AI models")
def list_ai_models():
    if not os.path.exists(MODEL_BASE_PATH):
        return JSONResponse(status_code=404, content={"models": [], "message": "Model base folder not found"})

    hdl_files = [f for f in os.listdir(MODEL_BASE_PATH) if f.endswith(".hdl")]
    models = [os.path.splitext(f)[0] for f in hdl_files]

    return {"models": models}
