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
        model_handle = ha.read_dl_model(model_file)
        dl_preprocess_param = create_dl_preprocess_param_from_model(model_handle,normalization_type="none",domain_handling="full_domain")      
        print(f"Model handle created: {dl_preprocess_param}")

        preprocess_dl_samples(dl_sample_batch, dl_preprocess_param)
        print(f"Samples preprocessed successfully, batch size: {len(dl_sample_batch)}")
        # results = ha.apply_dl_model(model_handle, dl_sample_batch)
        print(f"Model applied successfully:")

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

def create_dl_preprocess_param_from_model(dl_model_handle, normalization_type="none", domain_handling="full_domain"):
    model_type = ha.get_dl_model_param(dl_model_handle, "type")
    image_width = ha.get_dl_model_param(dl_model_handle, "image_width")
    image_height = ha.get_dl_model_param(dl_model_handle, "image_height")
    num_channels = ha.get_dl_model_param(dl_model_handle, "image_num_channels")
    range_min = ha.get_dl_model_param(dl_model_handle, "image_range_min")
    range_max = ha.get_dl_model_param(dl_model_handle, "image_range_max")
    
    print(f"model_type = {model_type}")
    print(f"image_width = {image_width}")
    print(f"image_height = {image_height}")
    print(f"num_channels = {num_channels}")
    print(f"range_min = {range_min}")
    print(f"range_max = {range_max}")

    ignore_class_ids = []
    set_background_id = []
    class_ids_background = []
    gen_param = ha.create_dict()

    print(f"Model type: {model_type}, Image size: {image_width}x{image_height}, Channels: {num_channels}")

    if model_type[0] == "detection":
        instance_type = ha.get_dl_model_param(dl_model_handle, "instance_type")
        is_instance_seg = ha.get_dl_model_param(dl_model_handle, "instance_segmentation")
        ha.set_dict_tuple(gen_param, "instance_type", "mask" if is_instance_seg == "true" else instance_type)

        if instance_type == "rectangle2":
            ignore_direction = ha.get_dl_model_param(dl_model_handle, "ignore_direction")
            if ignore_direction == "true":
                ha.set_dict_tuple(gen_param, "ignore_direction", True)
            elif ignore_direction == "false":
                ha.set_dict_tuple(gen_param, "ignore_direction", False)

            class_ids_no_orientation = ha.get_dl_model_param(dl_model_handle, "class_ids_no_orientation")
            ha.set_dict_tuple(gen_param, "class_ids_no_orientation", class_ids_no_orientation)

    elif model_type[0]  == "segmentation":
        ignore_class_ids = ha.get_dl_model_param(dl_model_handle, "ignore_class_ids")

    elif model_type[0]  == "3d_gripping_point_detection":
        num_channels = 1

    elif model_type[0]  in [
        "anomaly_detection", "gc_anomaly_detection",
        "classification", "multi_label_classification",
        "counting", "ocr_detection", "ocr_recognition"
    ]:
        pass  # ไม่มีอะไรต้องทำเพิ่ม

    else:
        raise Exception(f"Unsupported model type: {model_type}")

    # สร้าง DLPreprocessParam แล้ว return
    dl_preprocess_param = create_dl_preprocess_param(
        model_type,
        image_width,
        image_height,
        num_channels,
        range_min,
        range_max,
        normalization_type,
        domain_handling,
        ignore_class_ids,
        set_background_id,
        class_ids_background,
        gen_param
    )
    return dl_preprocess_param

def create_dl_preprocess_param(
    dl_model_type,
    image_width,
    image_height,
    image_num_channels,
    image_range_min,
    image_range_max,
    normalization_type,
    domain_handling,
    ignore_class_ids,
    set_background_id,
    class_ids_background,
    gen_param
):
    # สร้าง dictionary
    dl_preprocess_param = ha.create_dict()

    # Set core fields
    ha.set_dict_tuple(dl_preprocess_param, "model_type", dl_model_type)
    ha.set_dict_tuple(dl_preprocess_param, "image_width", image_width)
    ha.set_dict_tuple(dl_preprocess_param, "image_height", image_height)
    ha.set_dict_tuple(dl_preprocess_param, "image_num_channels", image_num_channels)

    # Set image range
    if len(image_range_min) == 0:
        ha.set_dict_tuple(dl_preprocess_param, "image_range_min", -127)
    else:
        ha.set_dict_tuple(dl_preprocess_param, "image_range_min", image_range_min)

    if len(image_range_max) == 0:
        ha.set_dict_tuple(dl_preprocess_param, "image_range_max", 128)
    else:
        ha.set_dict_tuple(dl_preprocess_param, "image_range_max", image_range_max)

    # Normalization & domain
    ha.set_dict_tuple(dl_preprocess_param, "normalization_type", normalization_type)
    ha.set_dict_tuple(dl_preprocess_param, "domain_handling", domain_handling)

    # Segmentation / 3D models
    if dl_model_type in ["segmentation", "3d_gripping_point_detection"]:
        ha.set_dict_tuple(dl_preprocess_param, "ignore_class_ids", ignore_class_ids)
        ha.set_dict_tuple(dl_preprocess_param, "set_background_id", set_background_id)
        ha.set_dict_tuple(dl_preprocess_param, "class_ids_background", class_ids_background)

    # Default augmentation
    ha.set_dict_tuple(dl_preprocess_param, "augmentation", "false")

    # Copy over generic parameters from GenParam if present
    if gen_param is not None:
        try:
            gen_param_names = ha.get_dict_param(gen_param, "keys", [])
            for key in gen_param_names:
                value = ha.get_dict_tuple(gen_param, key)
                ha.set_dict_tuple(dl_preprocess_param, key, value)
        except Exception as e:
            print(f"[WARNING] Could not copy gen_param: {e}")

    # Detection-specific defaults
    if dl_model_type == "detection":
        keys_exist = ha.get_dict_param(
            dl_preprocess_param,
            "key_exists",
            ["instance_type", "ignore_direction", "instance_segmentation"]
        )

        if not keys_exist[0]:
            ha.set_dict_tuple(dl_preprocess_param, "instance_type", "rectangle1")

        instance_type = ha.get_dict_tuple(dl_preprocess_param, "instance_type").s
        if instance_type == "rectangle2" and not keys_exist[1]:
            ha.set_dict_tuple(dl_preprocess_param, "ignore_direction", False)

        if keys_exist[2]:
            is_instance_seg = ha.get_dict_tuple(dl_preprocess_param, "instance_segmentation")
            is_instance_seg_str = str(is_instance_seg).lower()
            if is_instance_seg_str not in ["true", "false"]:
                raise Exception(f"Invalid generic parameter for 'instance_segmentation': {is_instance_seg}")
            if is_instance_seg_str == "true":
                ha.set_dict_tuple(dl_preprocess_param, "instance_type", "mask")

    return dl_preprocess_param

def preprocess_dl_samples(dl_sample_batch, dl_preprocess_param):
    # ✅ Copy dict เพื่อหลีกเลี่ยง race condition
    dl_preprocess_param = ha.copy_dict(dl_preprocess_param, [], [])

    # ✅ ตรวจสอบพารามิเตอร์ preprocess
    check_dl_preprocess_param(dl_preprocess_param)

    # ✅ วนลูปทุก sample
    for i in range(len(dl_sample_batch)):
        dl_sample = dl_sample_batch[i]

        # 1. Augmentation (หากมี)
        ha.preprocess_dl_model_augmentation_data(dl_sample, dl_preprocess_param)

        # 2. ตรวจสอบว่ามี key 'image'
        image_exists = ha.get_dict_param(dl_sample, "key_exists", "image")
        if not image_exists:
            raise Exception(f"Sample {i} missing required key 'image'")

        # 3. ดึงภาพออกมา
        image_raw = ha.get_dict_object(dl_sample, "image")

        # 4. Preprocess ภาพ
        image_preprocessed = ha.preprocess_dl_model_images(image_raw, dl_preprocess_param)

        # 5. ใส่กลับลง dict
        ha.set_dict_object(image_preprocessed, dl_sample, "image")

        # 6. ตรวจ key เฉพาะของ model type
        keys_exist = ha.get_dict_param(
            dl_sample,
            "key_exists",
            ["anomaly_ground_truth", "bbox_row1", "bbox_phi", "mask", "segmentation_image", "word"]
        )
        anomaly_exist = keys_exist[0]
        rect1_exist = keys_exist[1]
        rect2_exist = keys_exist[2]
        mask_exist = keys_exist[3]
        seg_exist = keys_exist[4]
        ocr_exist = keys_exist[5]

        # 7. Anomaly ground truth
        if anomaly_exist:
            anomaly_raw = ha.get_dict_object(dl_sample, "anomaly_ground_truth")
            anomaly_pre = ha.preprocess_dl_model_anomaly(anomaly_raw, dl_preprocess_param)
            ha.set_dict_object(anomaly_pre, dl_sample, "anomaly_ground_truth")

        # 8. Bounding box
        if rect1_exist:
            ha.preprocess_dl_model_bbox_rect1(image_raw, dl_sample, dl_preprocess_param)
        elif rect2_exist:
            ha.preprocess_dl_model_bbox_rect2(image_raw, dl_sample, dl_preprocess_param)

        # 9. Instance mask
        if mask_exist:
            ha.preprocess_dl_model_instance_masks(image_raw, dl_sample, dl_preprocess_param)

        # 10. Segmentation image
        if seg_exist:
            seg_raw = ha.get_dict_object(dl_sample, "segmentation_image")
            seg_pre = ha.preprocess_dl_model_segmentations(image_raw, seg_raw, dl_preprocess_param)
            ha.set_dict_object(seg_pre, dl_sample, "segmentation_image")

        # 11. OCR target
        if ocr_exist and rect2_exist:
            ha.gen_dl_ocr_detection_targets(dl_sample, dl_preprocess_param)

        # 12. 3D data
        key3d_exist = ha.get_dict_param(dl_sample, "key_exists", ["x", "y", "z", "normals"])
        if max(key3d_exist) == 1:
            image_domain = ha.get_domain(image_raw)
            for channel in ["x", "y", "z", "normals"]:
                ha.crop_dl_sample_image(image_domain, dl_sample, channel, dl_preprocess_param)
            ha.preprocess_dl_model_3d_data(dl_sample, dl_preprocess_param)

def check_dl_preprocess_param(dl_preprocess_param):
    # --- STEP 1: check_params flag ---
    check_flag = True
    if ha.get_dict_param(dl_preprocess_param, "key_exists", ["check_params"])[0]:
        check_flag = ha.get_dict_tuple(dl_preprocess_param, "check_params")
        if str(check_flag[0]).lower() in ["false", "0"]:
            return

    # --- STEP 2: model_type (must exist) ---
    try:
        model_type = str(ha.get_dict_tuple(dl_preprocess_param, "model_type")[0])
    except Exception:
        raise Exception("DLPreprocessParam needs the parameter: 'model_type'")

    supported_model_types = [
        'counting', '3d_gripping_point_detection', 'anomaly_detection', 'classification',
        'detection', 'gc_anomaly_detection', 'multi_label_classification',
        'ocr_recognition', 'ocr_detection', 'segmentation'
    ]
    if model_type not in supported_model_types:
        raise Exception(f"Unsupported model_type '{model_type}'")

    # --- STEP 3: Required keys ---
    required_general = [
        'model_type', 'image_width', 'image_height', 'image_num_channels',
        'image_range_min', 'image_range_max', 'normalization_type', 'domain_handling'
    ]
    required_seg = ['ignore_class_ids', 'set_background_id', 'class_ids_background']
    detection_optional = ['instance_type', 'ignore_direction', 'class_ids_no_orientation', 'instance_segmentation']

    keys_exist = ha.get_dict_param(dl_preprocess_param, "key_exists", required_general)
    for k, exists in zip(required_general, keys_exist):
        if not exists:
            raise Exception(f"DLPreprocessParam needs the parameter: '{k}'")

    # --- STEP 4: Validate known keys and values ---
    input_keys = ha.get_dict_param(dl_preprocess_param, "keys", [])
    for key in input_keys:
        value = ha.get_dict_tuple(dl_preprocess_param, key)
        if key == "normalization_type":
            if str(value[0]) not in ['all_channels', 'first_channel', 'constant_values', 'none']:
                raise Exception(f"Invalid value for '{key}': {value}")
        elif key == "domain_handling":
            valid_vals = ['full_domain', 'crop_domain']
            if model_type in ['anomaly_detection', 'gc_anomaly_detection', '3d_gripping_point_detection']:
                valid_vals.append('keep_domain')
            if str(value[0]) not in valid_vals:
                raise Exception(f"Invalid value for '{key}': {value}")
        elif key == "augmentation":
            if str(value[0]).lower() not in ["true", "false"]:
                raise Exception(f"Invalid value for 'augmentation': {value}")

    # --- STEP 5: Model-specific validation ---
    if model_type in ["segmentation", "3d_gripping_point_detection"]:
        for key in detection_optional:
            if ha.get_dict_param(dl_preprocess_param, "key_exists", [key])[0]:
                val = ha.get_dict_tuple(dl_preprocess_param, key)
                if len(val) > 0:
                    raise Exception(f"'{key}' should be [] for model type '{model_type}'")

        set_bg_id = []
        class_ids_bg = []
        ignore_ids = []
        if ha.get_dict_param(dl_preprocess_param, "key_exists", ["set_background_id"])[0]:
            set_bg_id = ha.get_dict_tuple(dl_preprocess_param, "set_background_id")
        if ha.get_dict_param(dl_preprocess_param, "key_exists", ["class_ids_background"])[0]:
            class_ids_bg = ha.get_dict_tuple(dl_preprocess_param, "class_ids_background")
        if ha.get_dict_param(dl_preprocess_param, "key_exists", ["ignore_class_ids"])[0]:
            ignore_ids = ha.get_dict_tuple(dl_preprocess_param, "ignore_class_ids")

        if model_type == "3d_gripping_point_detection":
            if len(set_bg_id) > 0 or len(class_ids_bg) > 0 or len(ignore_ids) > 0:
                raise Exception("set_background_id, class_ids_background, and ignore_class_ids should all be [] for 3d_gripping_point_detection")

        if (len(set_bg_id) > 0 and len(class_ids_bg) == 0) or (len(class_ids_bg) > 0 and len(set_bg_id) == 0):
            raise Exception("Both 'set_background_id' and 'class_ids_background' must be set together")

        if len(set_bg_id) > 1:
            raise Exception("Only one class ID allowed in 'set_background_id'")

        if len(set(set_bg_id) & set(class_ids_bg)) > 0:
            raise Exception("Overlap between 'set_background_id' and 'class_ids_background' is not allowed")

    elif model_type == "detection":
        for key in required_seg:
            if ha.get_dict_param(dl_preprocess_param, "key_exists", [key])[0]:
                val = ha.get_dict_tuple(dl_preprocess_param, key)
                if len(val) > 0:
                    raise Exception(f"'{key}' should be [] for detection model")

        optional_keys = ha.get_dict_param(dl_preprocess_param, "key_exists", detection_optional)
        if optional_keys[0]:
            inst_type = str(ha.get_dict_tuple(dl_preprocess_param, "instance_type")[0])
            if inst_type not in ["rectangle1", "rectangle2", "mask"]:
                raise Exception(f"Invalid 'instance_type': {inst_type}")
        if optional_keys[3]:
            seg = str(ha.get_dict_tuple(dl_preprocess_param, "instance_segmentation")[0]).lower()
            if seg not in ["true", "false"]:
                raise Exception(f"Invalid 'instance_segmentation': {seg}")
        if optional_keys[1]:
            ignore_dir = str(ha.get_dict_tuple(dl_preprocess_param, "ignore_direction")[0]).lower()
            if ignore_dir not in ["true", "false"]:
                raise Exception(f"Invalid 'ignore_direction': {ignore_dir}")
        if optional_keys[2]:
            no_orient = ha.get_dict_tuple(dl_preprocess_param, "class_ids_no_orientation")
            if len(no_orient) > 0 and not all(int(v) >= 0 for v in no_orient):
                raise Exception("class_ids_no_orientation must contain only non-negative integers")

    # --- STEP 6: Fixed image range check ---
    if model_type in ["classification", "multi_label_classification", "detection"]:
        if ha.get_dict_param(dl_preprocess_param, "key_exists", ["image_range_min"])[0]:
            if ha.get_dict_tuple(dl_preprocess_param, "image_range_min")[0] != -127:
                raise Exception(f"For model type {model_type} image_range_min must be -127")
        if ha.get_dict_param(dl_preprocess_param, "key_exists", ["image_range_max"])[0]:
            if ha.get_dict_tuple(dl_preprocess_param, "image_range_max")[0] != 128:
                raise Exception(f"For model type {model_type} image_range_max must be 128")


