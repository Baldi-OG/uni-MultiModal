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
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import base64
import io
from PIL import Image
import open_clip

app = FastAPI(title="Image Classification API", version="v1.0")

##
# To execute run this:
# uv run server.py --port 8000 --weights-dir ./weights
##

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


### Classes ###

class ErrorBase(BaseModel):
    error: str


class BadRequest(ErrorBase):
    code: int = 400


class PayloadTooLarge(ErrorBase):
    code: int = 413


class InternalServerError(ErrorBase):
    code: int = 500


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
          response_model=PredictionResponse,
          responses=PREDICT_RESPONSES)
def predict(body: PredictionRequest):
    try:
        image, format = image_from_base64(body.image)
    except ValueError as e:
        return BadRequest(error=str(e)) # directly return base64 decoding errors as BadRequest

    if not validate_image(image, format):
        return BadRequest(error="Image must be a PNG and between 1x1 and 1024x1024 pixels")
    
    end_points = model.predict(image, body.classes)
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



def image_from_base64(image_64: str):
    # Accept both raw base64 and data URLs
    if image_64.startswith("data:"):
        header, _, encoded = image_64.partition(",")
        if not encoded:
            raise ValueError("Empty base64 payload")
        image_64 = encoded

    try:
        img_bytes = base64.b64decode(image_64, validate=True)
    except (ValueError, binascii.Error) as e:
        raise ValueError("Invalid base64 data") from e

    # PNG detection: check signature before decoding
    if not img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Image is not a valid PNG")

    with Image.open(io.BytesIO(img_bytes)) as img:
        img.load()  # fully decode image data
        return img.copy(), img.format  # return a usable in-memory image


def validate_image(image: Image.Image, format) -> bool:
    """
    Returns True if the image fulfills the requirements, False otherwise.
    """
    # verify image is png
    if format != "PNG":
        return False

    #verify image is between 1x1 and 1024x1024 pixels
    if image.width < 1 or image.height < 1 or image.width > 1024 or image.height > 1024:
        return False

    return True

#TODO: images need to be properly preprocessed (transformed to RGB with no alpha channel)


class ClipModel:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
        self.emotion_classes = self.tokenizer(self.emotions)
        if torch.cuda.is_available():
            self.emotion_classes = self.emotion_classes.cuda()
        #TODO: train emotion classifier head

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:

        return self.preprocess(image).unsqueeze(0)  # Add batch dimension

    @torch.no_grad()
    def predict(self, image: Image.Image, classes: list[str]) -> dict[str, dict[str, float]]:
        # TODO: at least from mnist tests the clip preprocessing seems to be worse, test whether this is true in general
        image_tensor = self.preprocess_image(image)  # Add batch dimension #TODO: is this necessary?
        text_inputs = self.tokenizer(classes)

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            text_inputs = text_inputs.cuda()

        image_features = self.model.encode_image(image_tensor)
        text_features = self.model.encode_text(text_inputs)
        emotion_features = self.model.encode_text(self.emotion_classes)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        emotion_features /= emotion_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        text_similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        emotion_similarity = (100.0 * image_features @ emotion_features.T).softmax(dim=-1)

        end_points = {
                "classes": {cls: float(text_similarity[0][i]) for i, cls in enumerate(classes)},
                "emotions": {cls: float(emotion_similarity[0][i]) for i, cls in enumerate(self.emotions)}
                }
        # Convert to dictionary
        return end_points


### Util ###
def get_model_version() -> str:
    # TODO: Grab this automatically from the git log
    return "last model-relevant git commit hash"


model = ClipModel()

if __name__ == '__main__':
    port = int(sys.argv[2])
    weights_dir = sys.argv[4]

    uvicorn.run(app="server:app", host="0.0.0.0", port=port, reload=True)
