import base64
import requests
import cv2
import numpy as np
import os

# ====== Set values here ======
API_URL = "http://127.0.0.1:8000/query-base64"
IMAGE_FOLDER = r"Images\Pill_crack"
MODEL_NAME = "segment_pill_defects"
CLIENT_ID = "Client123"
# ===========================

# Allowed image extensions
allowed_exts = (".jpg", ".jpeg", ".png", ".bmp")

# Get sorted list of image files in folder
image_files = sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(allowed_exts)]
)

if not image_files:
    raise FileNotFoundError(f"No images found in folder: {IMAGE_FOLDER}")

for img_name in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    print(f"\nProcessing: {img_path}")

    # Read the image file and convert it to Base64
    with open(img_path, "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {"image_base64": image_base64, "model": MODEL_NAME, "client_id": CLIENT_ID}

    # call API
    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()
    result = resp.json()

    print("Status:", result.get("Status_Code"), result.get("Status_Message"))
    print("Processing Time:", result.get("Processing_Time"), "ms")
    print("Objects detected:", len(result.get("DataInfo", [])))
    print(f" Detected classes: {[item.get('Bbox_Class_Name') for item in result.get('DataInfo', [])]}")

    # --- Prepare images ---
    orig = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if orig is None:
        raise RuntimeError("Failed to decode original image bytes")

    H, W = orig.shape[:2]
    overlay = orig.copy()

    # Deterministic color palette (BGR)
    palette = [
        (255, 0, 0),     # blue
        (0, 255, 0),     # green
        (0, 0, 255),     # red
        (255, 255, 0),   # cyan-ish
        (255, 0, 255),   # magenta
        (0, 255, 255),   # yellow
        (128, 0, 255),   # purple-ish
        (0, 128, 255),   # orange-ish
    ]

    alpha = 0.45  # Opacity of the overlay color

    # Combine overlay from all masks (use only pixels with value == 255)
    for idx, item in enumerate(result.get("DataInfo", [])):
        mask_b64 = item.get("Mask_image_base64")
        if not mask_b64:
            continue

        mask_bytes = base64.b64decode(mask_b64)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Resize to match the original image (as a safeguard)
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # *** Use only the areas where pixel value == 255 ***
        mask_255 = (mask == 255)

        if not np.any(mask_255):
            continue

        color = palette[idx % len(palette)]
        color_layer = np.zeros_like(orig)
        color_layer[mask_255] = color

        # Blend only the areas where mask == 255
        overlay[mask_255] = cv2.addWeighted(color_layer[mask_255], alpha,
                                            overlay[mask_255], 1 - alpha, 0)

    # Display only 2 windows: Original + Overlay
    cv2.imshow("Original", orig)
    cv2.imshow("Overlay (threshold==255 only)", overlay)
    print("Press any key in an image window to proceed to the next image...")


    # Wait for 0.2 seconds before showing the next image
    # (Previously: waited indefinitely for key press, with ESC to exit)
    # key = cv2.waitKey(0)
    # if key == 27:  # ESC key to exit early
    #     break
    cv2.waitKey(200)  # Wait 0.2 seconds before next image

cv2.destroyAllWindows()
