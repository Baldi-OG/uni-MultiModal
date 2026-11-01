# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch>=2.14.1",
#   "torchaudio>=2.11.0",
#   "torchvision>=0.29.1",
#   "fastapi==0.162.0",
#   "uvicorn==0.34.0",
#   "pydantic>=2.14.2",
#   "open-clip-torch>=3.3.0",
# ]
# ///
import base64
import io
import sys

import numpy as np
import open_clip
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from numpy.typing import NDArray
from pydantic import BaseModel
import argparse

app = FastAPI(title="Image Classification API", version="v1.0")

##
# To execute run this:
# uv run server.py --port 8000 --weights-dir ./weights
##

# noinspection bad-argument-type
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


### Models ###

class PredictionRequest(BaseModel):
    image: str
    classes: list[str]


class PredictionResponse(BaseModel):
    classes: dict[str, float]
    emotions: dict[str, float]


class InfoResponse(BaseModel):
    team: str = "team-01"
    model_version: str
    max_image_size: list[int] = [1024, 1024]
    max_classes: int = 50


### Constants ###

PREDICT_RESPONSES = {
    200: {"description": "OK",
          "model": PredictionResponse,
          "content": {
              "application/json": {
                  "example": {
                      "classes": {
                          "cat": 0.71,
                          "dog": 0.22,
                          "a person at a desk": 0.07
                      },
                      "emotions": {
                          "anger": 0.02,
                          "disgust": 0.01,
                          "fear": 0.03,
                          "happiness": 0.81,
                          "sadness": 0.04,
                          "surprise": 0.05,
                          "neutral": 0.04
                      }
                  }
              }
          }},
    400: {
        "description": "Malformed JSON, missing or wrongly typed field, invalid base64, image is not a PNG, empty or"
                       " duplicate class strings, class longer than 200 characters, more than 50 classes"},
    413: {"description": "Image larger than 1024×1024 px, or request body larger than 10 MB"},
    500: {"description": "Unexpected internal failure"}
}

INFO_RESPONSES = {
    200: {"description": "OK",
          "content": {
              "application/json": {
                  "example": {
                      "team": "team-03",
                      "model_version": "a1b2c3d",
                      "max_image_size": [1024, 1024],
                      "max_classes": 50
                  }
              }
          }},
    500: {"description": "Unexpected internal failure"}
}


## Endpoints ###

@app.post("/predict",
          responses=PREDICT_RESPONSES
          )
def predict(body: PredictionRequest):
    # convert and validate base64 image
    try:
        image, image_format = image_from_base64(body.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # directly return base64 decoding errors as BadRequest
    if image.__sizeof__() > 1e7:  # Larger than 10MB
        raise HTTPException(status_code=413, detail="Image must not be larger than 10MB")
    if image_format != "PNG":
        raise HTTPException(status_code=400, detail="Image must be a PNG")
    if image.width < 1 or image.height < 1:
        raise HTTPException(status_code=400, detail="Image must be at lest 1x1 pixels large")
    if image.width > 1024 or image.height > 1024:
        raise HTTPException(status_code=413, detail="Image must not be larger than 1024x1024 pixels")

    # validate classes
    classes = np.unique(np.array(body.classes))  # filter duplicates and transform to tensor
    # duplicates are not valid so return error
    if classes.shape[0] != len(body.classes):
        raise HTTPException(status_code=400, )
    if classes.shape[0] > 50:
        raise HTTPException(status_code=400, detail="Too many classes, maximum is 50")
    if any(len(cls) > 200 for cls in body.classes):
        raise HTTPException(status_code=400, detail="Class strings must be at most 200 characters long")

    end_points = model.predict(image, classes)
    return PredictionResponse(
        **end_points
    )


@app.get("/info",
         response_model=InfoResponse,
         responses=INFO_RESPONSES)
def info():
    return InfoResponse(
        model_version=get_model_version()
    )


def image_from_base64(image_64: str) -> tuple[Image.Image, str]:
    # Accept both raw base64 and data URLs
    if image_64.startswith("data:"):
        header, _, encoded = image_64.partition(",")
        if not encoded:
            raise ValueError("Empty base64 payload")
        image_64 = encoded

    try:
        img_bytes = base64.b64decode(image_64, validate=True)
    except ValueError as e:
        raise ValueError("Invalid base64 data") from e

    # PNG detection: check signature before decoding
    if not img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Image is not a valid PNG")

    with Image.open(io.BytesIO(img_bytes)) as img:
        img.load()  # fully decode image data
        return img.copy(), str(img.format)  # return a usable in-memory image


class ClipModel:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.clip_backbone, _, self.preprocess = open_clip.create_model_and_transforms(model_name=model_name,
                                                                               pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
        self.emotion_classes = self.tokenizer(self.emotions)

        self.emotion_classifier_head = torch.nn.Linear(self.clip_backbone.visual.output_dim, self.clip_backbone.visual.output_dim)

        if torch.cuda.is_available():
            self.clip_backbone.cuda()
            self.emotion_classifier_head.cuda()
            self.emotion_classes = self.emotion_classes.cuda()

        self.clip_backbone.eval()
        self.emotion_classifier_head.eval()

        self.emotion_features = self.clip_backbone.encode_text(self.emotion_classes)
        self.emotion_features /= self.emotion_features.norm(dim=-1, keepdim=True)

        try:
            self.emotion_classifier_head.load_state_dict(torch.load("weights/emotion_classifier_head.pth"))
        except FileNotFoundError:
            print("Warning: Could not find weights for emotion classifier head. Please ensure that the file 'weights/emotion_classifier_head.pth' exists.", file=sys.stderr)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        # remove possible alpha channel and convert to 8-bit RGB
        image = image.convert("RGB")

        return self.preprocess(image).unsqueeze(0)  # Add batch dimension

    @torch.no_grad()
    def predict(self, image: Image.Image, classes: NDArray) -> dict[str, dict[str, float]]:
        # TODO: at least from mnist tests the clip preprocessing seems to be worse, test whether this is true in general
        image_tensor = self.preprocess_image(image)  # Add batch dimension #TODO: is this necessary?
        text_inputs = self.tokenizer(classes)

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            text_inputs = text_inputs.cuda()

        image_features = self.clip_backbone.encode_image(image_tensor)
        image_emotion_features = self.emotion_classifier_head(image_features)
        text_features = self.clip_backbone.encode_text(text_inputs)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_emotion_features /= image_emotion_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        text_similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        emotion_similarity = (100.0 * image_emotion_features @ self.emotion_features.T).softmax(dim=-1)

        end_points = {
            "classes": {cls: float(text_similarity[0][i]) for i, cls in enumerate(classes)},
            "emotions": {cls: float(emotion_similarity[0][i]) for i, cls in enumerate(self.emotions)}
        }
        # Convert to dictionary
        return end_points


### Util ###
def get_model_version() -> str:
    # TODO: Grab this automatically from the git log
    return "a4e3d877e8786b423870c01bfe018c4715ee39d7"


model = ClipModel()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--random_seed", default=671611312, type=int, help="random seed")
    parser.add_argument("-p", "--port", default=8000, type=int, help="port to run the server on")
    parser.add_argument("-w", "--weights-dir", default="./weights", type=str, help="directory to load model weights from")

    args = parser.parse_args()
    port = args.port
    weights_dir = args.weights_dir

    uvicorn.run(app="server:app", host="0.0.0.0", port=port, reload=True)
    torch.random.manual_seed(671611312)    # set fixed seed to ensure determinism requirement
