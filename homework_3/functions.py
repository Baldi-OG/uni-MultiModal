import base64
import io
import json
from hashlib import md5
from typing import Optional, Dict

import requests
import torch
from PIL import Image

session = requests.Session()

TARGET_SERVER_URL = "https://wstud-mmml26-classification.web.webis.de"
RESPONSE_CACHE_FILE = "../cache/responses.json"


### Parsing ###

def parse_img(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_base64_bytes = base64.b64encode(buf.getvalue())
    img_base64 = img_base64_bytes.decode("utf-8")
    img_hash = md5(img_base64_bytes).hexdigest()
    return img_base64, img_hash

def api_response_to_tensor(api_response: Dict) -> torch.Tensor:
    return torch.Tensor([value for value in api_response['classes'].values()])

### Cache ###

def get_cache(img_hash) -> Optional[Dict]:
    with open(RESPONSE_CACHE_FILE, "r") as f:
        cache_content = json.load(f)
        return cache_content.get(img_hash, None)


def set_cache(img_hash, api_response) -> None:
    with open(RESPONSE_CACHE_FILE, "r") as f:
        cache_content = json.load(f)
        cache_content[img_hash] = api_response
    with open(RESPONSE_CACHE_FILE, "w") as f:
        json.dump(cache_content, f)


### Network ###

def query_api(img_base64) -> Dict:
    try:
        resp = session.post(
            TARGET_SERVER_URL + "/predict",
            json={"image": img_base64, "classes": ["class1", "class2", "class3", "class4"]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Request failed {e}")
        return {}
