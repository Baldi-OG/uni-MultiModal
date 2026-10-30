# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "fastapi==0.162.0",
#   "uvicorn==0.34.0",
#   "pydantic>=2.14.2",
# ]
# ///
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    # TODO Implement this
    return PredictionResponse(
        classes={
        },
        emotions={

        }
    )


@app.get("/info",
         response_model=InfoResponse,
         responses=INFO_RESPONSES)
def info():
    return InfoResponse(
        model_version=get_model_version()
    )


### Util ###
def get_model_version() -> str:
    # TODO: Grab this automatically from the git log
    return "last model-relevant git commit hash"


if __name__ == '__main__':
    port = int(sys.argv[2])
    weights_dir = sys.argv[4]

    uvicorn.run(app="server:app", host="0.0.0.0", port=port, reload=True)
