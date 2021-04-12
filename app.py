import os
import uuid
import logging

from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import constants

from score import Predict

some_file_path = "12283150_12d37e6389_z.jpg"

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class FormData(BaseModel):
    name: str
    _file: UploadFile = File(...)

app = FastAPI()


@app.get("/image")
async def return_image():
    return FileResponse(some_file_path)

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):

    contents = await file.read()  # <-- Important!
    path = f"{constants.INPUT_DATA}/{file.filename}"

    # example of how you can save the file
    with open(path, "wb") as f:
        f.write(contents)
    
    # prediction = Predict(file.filename, ["balloon"])
    # try:
    #     stored_path = prediction.predict()
    # except Exception as e:
    #     raise e

    return FileResponse(path)
    # return {"file": FileResponse(path)} returns filepath


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}