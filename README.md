# HALCON Deep Learning FastAPI Service

This FastAPI app provides endpoints for image inference using HALCON deep learning models.  
It supports image file upload, base64 input, and returns prediction results with optional mask preview.

## Requirements

- Python 3.8+
- [HALCON 22.11 Progress Edition](https://www.mvtec.com/products/halcon) or newer (must include Deep Learning components)
- HALCON must be properly installed and licensed on the machine
- `halcon` Python bindings must be available (usually via `pip install halcon` from MVTec installation)


## Setup Instructions

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. Run the API server
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. **Open Swagger UI**
```bash
http://localhost:8000/docs
```