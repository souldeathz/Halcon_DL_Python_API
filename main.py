from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from datetime import datetime
import base64
import os
import time
from PIL import Image
import io
import json

app = FastAPI()

with open("config.json", "r") as f:
    config = json.load(f)

MODEL_BASE_PATH = config["MODEL_BASE_PATH"]

class DataInfoItem(BaseModel):
    Bbox_Class_ID: int
    Bbox_Class_Name: str
    Bbox_Confidence: float
    Bbox_Row1: float
    Bbox_Col1: float
    Bbox_Row2: float
    Bbox_Col2: float
    Mask_image_base64: str

class QueryResponse(BaseModel):
    Status_Code: str
    Status_Message: str
    DataInfo: List[DataInfoItem]
    Processing_Time: str
    Client_ID: str
    Timestamp: str


@app.get("/model-list", summary="List available AI models")
def list_ai_models():
    if not os.path.exists(MODEL_BASE_PATH):
        return JSONResponse(status_code=404, content={"models": [], "message": "Model base folder not found"})

    # ค้นหาเฉพาะ .hdl ในโฟลเดอร์หลัก
    hdl_files = [f for f in os.listdir(MODEL_BASE_PATH) if f.endswith(".hdl")]

    # ตัดนามสกุลออก ให้ชื่อ model เป็นแค่ชื่อไฟล์
    models = [os.path.splitext(f)[0] for f in hdl_files]

    return {"models": models}

@app.post("/query-upload", response_model=QueryResponse, summary="Upload image and get result")
async def query_upload(
    image_file: UploadFile = File(..., description="Input image file (.jpg, .png, etc.)"),
    model: str = Form(..., description="Model name (e.g., 'model' or 'segment_v2')"),
    client_id: str = Form(..., description="Client ID (e.g., 'Client123')")
):
    start_time = time.time()

    # 1. อ่านและตรวจสอบภาพ
    try:
        image_bytes = await image_file.read()
        Image.open(io.BytesIO(image_bytes)).verify()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    except Exception:
        return JSONResponse(
            status_code=200,
            content={
                "Status_Code": "01",
                "Status_Message": "Invalid image format or decode error",
                "DataInfo": [],
                "Processing_Time": "0",
                "Client_ID": client_id,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )

    # 2. ตรวจสอบว่ามีไฟล์ .hdl ตรงชื่อ model ไหม
    model_file = os.path.join(MODEL_BASE_PATH, model + ".hdl")
    if not os.path.isfile(model_file):
        return JSONResponse(
            status_code=200,
            content={
                "Status_Code": "02",
                "Status_Message": f"Model '{model}' not found or missing .hdl file",
                "DataInfo": [],
                "Processing_Time": "0",
                "Client_ID": client_id,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )

    # 3. Success return
    processing_time_ms = (time.time() - start_time) * 1000

    return {
        "Status_Code": "OK",
        "Status_Message": "",
        "DataInfo": [],
        "Processing_Time": f"{processing_time_ms:.4f}",
        "Client_ID": client_id,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
