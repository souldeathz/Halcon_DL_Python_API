from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from typing import List
import base64
import os
from models import QueryResponse, Base64ImageRequest
from inference_service import process_inference, mask_store

app = FastAPI()

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
                "Timestamp": "",
                "DataInfo": []
            }
        )

    image_bytes = await image_file.read()
    return process_inference(image_bytes, model, client_id, file_ext, include_preview_url=True)

# -------- Endpoint 2: Upload Base64 --------
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
                "Timestamp": "",
                "DataInfo": []
            }
        )

    return process_inference(image_bytes, request.model, request.client_id, file_ext=".jpg", include_preview_url=False)

# -------- Endpoint 3: Preview mask --------
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

# -------- Endpoint 4: List available models --------
@app.get("/model-list", summary="List available AI models")
def list_ai_models():
    from inference_service import MODEL_BASE_PATH
    if not os.path.exists(MODEL_BASE_PATH):
        return JSONResponse(status_code=404, content={"models": [], "message": "Model base folder not found"})

    hdl_files = [f for f in os.listdir(MODEL_BASE_PATH) if f.endswith(".hdl")]
    models = [os.path.splitext(f)[0] for f in hdl_files]
    return {"models": models}
