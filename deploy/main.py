from flask import Flask, jsonify, request
import requests
import torch
from loguru import logger
from PIL import Image
import base64
import numpy as np
from io import BytesIO
import re

app = Flask(__name__)

def base64toimg(data):
    pil_img = Image.open(BytesIO(base64.b64decode(base64)))
    img = np.array(pil_img)
    return img

@app.route("/")
@app.route("/index")
def index():
    resp = {
        "hello": "world"
    }
    return jsonify(resp), 200

@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Processing predict request")
    body = request.json
    base64 = body["base64"]
    img = base64toimg(base64)
    logger.info(img.shape)
    resp = {
        "resp": "processed"
    }
    return jsonify(resp), 200


if __name__ == "__main__":
    logger.info("...Service started...")
    app.run(host="0.0.0.0", port=9090)