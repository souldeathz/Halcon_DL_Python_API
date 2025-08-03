# inference_service.py

import os
import time
import uuid
import base64
import traceback
import json
from datetime import datetime
from fastapi.responses import JSONResponse
import halcon as ha
from PIL import Image
import io
from models import ExpiringDict

# ------------------ CONFIG & PATH SETUP ------------------

base_dir = os.path.dirname(os.path.abspath(__file__))
TEMP_FOLDER = os.path.join(base_dir, "Temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

config_path = os.path.join(base_dir, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

MODEL_BASE_PATH = config["MODEL_BASE_PATH"]

# ------------------ HALCON PROCEDURE INIT ------------------

hdev_path = os.path.join(base_dir, 'Engine', 'Process.hdev')
program = ha.HDevProgram(hdev_path)

proc_preprocess_dl_samples = ha.HDevProcedure.load_local(program, 'preprocess_dl_samples')
proc_preprocess_dl_samples_call = ha.HDevProcedureCall(proc_preprocess_dl_samples)

create_dl_preprocess_param_from_model = ha.HDevProcedure.load_local(program, 'create_dl_preprocess_param_from_model')
create_dl_preprocess_param_from_model_call = ha.HDevProcedureCall(create_dl_preprocess_param_from_model)

proc_gen_dl_samples_from_images = ha.HDevProcedure.load_local(program, 'gen_dl_samples_from_images')
proc_gen_dl_samples_from_images_call = ha.HDevProcedureCall(proc_gen_dl_samples_from_images)

proc_Sort_Segmentation_Obj = ha.HDevProcedure.load_local(program, 'Sort_Segmentation_Obj')
proc_Sort_Segmentation_Obj_call = ha.HDevProcedureCall(proc_Sort_Segmentation_Obj)

# ------------------ SHARED CACHE ------------------

model_cache = {}
mask_store = ExpiringDict()

# ------------------ MAIN FUNCTION ------------------

def process_inference(image_bytes: bytes, model: str, client_id: str, file_ext: str, include_preview_url: bool = True):
    start_time = time.time()

    try:
        Image.open(io.BytesIO(image_bytes)).verify()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    except Exception:
        return _error_response("01", "Invalid image format or decode error", client_id)

    model_file = os.path.join(MODEL_BASE_PATH, model + ".hdl")
    if not os.path.isfile(model_file):
        return _error_response("02", f"Model '{model}' not found or missing .hdl file", client_id)

    temp_filename = f"input_{uuid.uuid4().hex[:8]}{file_ext}"
    temp_path = os.path.join(TEMP_FOLDER, temp_filename)
    with open(temp_path, "wb") as f:
        f.write(image_bytes)

    try:
        ho_image = ha.read_image(temp_path)

        proc_gen_dl_samples_from_images_call.set_input_iconic_param_by_name("Images", ho_image)
        proc_gen_dl_samples_from_images_call.execute()
        dl_sample_batch = proc_gen_dl_samples_from_images_call.get_output_control_param_by_name("DLSampleBatch")

        if model_file in model_cache:
            model_handle = model_cache[model_file]
        else:
            model_handle = ha.read_dl_model(model_file)
            model_cache[model_file] = model_handle

        create_dl_preprocess_param_from_model_call.set_input_control_param_by_name("DLModelHandle", model_handle)
        create_dl_preprocess_param_from_model_call.set_input_control_param_by_name("NormalizationType", "none")
        create_dl_preprocess_param_from_model_call.set_input_control_param_by_name("DomainHandling", "full_domain")
        create_dl_preprocess_param_from_model_call.set_input_control_param_by_name("SetBackgroundID", [])
        create_dl_preprocess_param_from_model_call.set_input_control_param_by_name("ClassIDsBackground", [])
        create_dl_preprocess_param_from_model_call.set_input_control_param_by_name("GenParam", ha.create_dict())
        create_dl_preprocess_param_from_model_call.execute()

        dl_preprocess_param = create_dl_preprocess_param_from_model_call.get_output_control_param_by_name("DLPreprocessParam")

        proc_preprocess_dl_samples_call.set_input_control_param_by_name("DLSampleBatch", dl_sample_batch)
        proc_preprocess_dl_samples_call.set_input_control_param_by_name("DLPreprocessParam", dl_preprocess_param)
        proc_preprocess_dl_samples_call.execute()

        results = ha.apply_dl_model(model_handle, dl_sample_batch, [])

        print(f"Model '{model}' applied successfully.")

        width, height = ha.get_image_size(ho_image)
        print(f"Image size: {width}x{height}")
        dl_image_width = ha.get_dl_model_param(model_handle, 'image_width')[0]
        dl_image_height = ha.get_dl_model_param(model_handle, 'image_height')[0]

        # Image_const = ha.gen_image_const("byte", width[0], height[0])
        print(f"DL Model size: {dl_image_width}x{dl_image_height}")
        zoom_w = width[0] / dl_image_width
        zoom_h = height[0] / dl_image_height
        print(f"Zoom factors: {zoom_w}, {zoom_h}")

        proc_Sort_Segmentation_Obj_call.set_input_control_param_by_name("DictHandle", results)
        proc_Sort_Segmentation_Obj_call.set_input_control_param_by_name("zoom_image_factor_width", zoom_w)
        proc_Sort_Segmentation_Obj_call.set_input_control_param_by_name("zoom_image_factor_height", zoom_h)
        proc_Sort_Segmentation_Obj_call.set_input_control_param_by_name("image_width", width[0])
        proc_Sort_Segmentation_Obj_call.set_input_control_param_by_name("image_height", height[0])

        proc_Sort_Segmentation_Obj_call.execute()

        DLResult = proc_Sort_Segmentation_Obj_call.get_output_control_param_by_name("out_DictHandle_all_object_detected")

        try:
            num_obj = len(DLResult)
        except Exception:
            num_obj = 0

        print(f"Number of detected objects: {num_obj}")

        result_list = []
        for i in range(num_obj):
            try:
                dict_selected = DLResult[i]

                # เขียน mask image ลง temp แล้ว encode base64
                mask_image = ha.get_dict_object(dict_selected, 'mask_image')
                mask_filename = f"mask_{uuid.uuid4().hex[:8]}.png"
                mask_path = os.path.join(TEMP_FOLDER, mask_filename)
                ha.write_image(mask_image, "png", 0, mask_path)

                with open(mask_path, "rb") as f:
                    mask_base64 = base64.b64encode(f.read()).decode("utf-8")

                try:
                    os.remove(mask_path)
                except Exception:
                    pass

                item = {
                    "Bbox_Class_ID": int(ha.get_dict_tuple(dict_selected, 'bbox_class_id')[0]),
                    "Bbox_Class_Name": ha.get_dict_tuple(dict_selected, 'bbox_class_name')[0],
                    "Bbox_Confidence": float(ha.get_dict_tuple(dict_selected, 'bbox_confidence')[0]),
                    "Bbox_Row1": float(ha.get_dict_tuple(dict_selected, 'bbox_row1')[0]),
                    "Bbox_Col1": float(ha.get_dict_tuple(dict_selected, 'bbox_col1')[0]),
                    "Bbox_Row2": float(ha.get_dict_tuple(dict_selected, 'bbox_row2')[0]),
                    "Bbox_Col2": float(ha.get_dict_tuple(dict_selected, 'bbox_col2')[0]),
                    "Mask_image_base64": mask_base64
                }

                if include_preview_url:
                    mask_id = f"mask_{uuid.uuid4().hex[:8]}"
                    mask_store[mask_id] = mask_base64
                    item["Mask_image_preview_url"] = f"http://127.0.0.1:8000/mask-preview/{mask_id}"

                result_list.append(item)

            except Exception as e:
                print(f"[WARN] Error parsing object #{i}: {e}")
                continue

    except Exception as e:
        return _error_response("99", f"HALCON Error: {str(e)}\nTraceback:\n{traceback.format_exc()}", client_id)

    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

    processing_time_ms = (time.time() - start_time) * 1000

    return {
        "Status_Code": "OK",
        "Status_Message": "",
        "Processing_Time": round(processing_time_ms, 4),
        "Client_ID": client_id,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "DataInfo": result_list
    }

# ------------------ HELPER ------------------

def _error_response(code, message, client_id):
    return JSONResponse(
        status_code=200,
        content={
            "Status_Code": code,
            "Status_Message": message,
            "Processing_Time": 0,
            "Client_ID": client_id,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "DataInfo": []
        }
    )
