import os
import uuid
import logging

from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import mrcnn.constants as constants

from mrcnn.score import Predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/health")
async def return_image():
    return {"health": "This app is healthy."}

@app.post("/v1/detect")
async def create_upload_file(file: UploadFile = File(...)):

    contents = await file.read()
    path = f"{constants.INPUT_DATA}/{file.filename}"

    with open(path, "wb") as f:
        f.write(contents)
    
    prediction = Predict(file.filename, ["balloon"])
    try:
        stored_path = prediction.predict()
    except Exception as e:
        raise e

    return FileResponse(path)
    # return {"file": FileResponse(path)} returns filepath
