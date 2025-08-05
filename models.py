from collections import OrderedDict
import time
from pydantic import BaseModel
from typing import List

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


class DataInfoItem(BaseModel):
    Bbox_Class_ID: int
    Bbox_Class_Name: str
    Mask_image_base64: str
    Mask_image_preview_url: str = None


class QueryResponse(BaseModel):
    Status_Code: str
    Status_Message: str
    DataInfo: List[DataInfoItem]
    Processing_Time: float
    Client_ID: str
    Timestamp: str

class Base64ImageRequest(BaseModel):
    image_base64: str
    model: str
    client_id: str