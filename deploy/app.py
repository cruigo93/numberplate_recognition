from flask import Flask, jsonify, request
import requests
import torch
from loguru import logger
from PIL import Image
import base64
import numpy as np
from io import BytesIO
from cfg import init_model
from transforms import get_valid_transforms
from pickle import load


app = Flask(__name__)
logger.info("...Service started...")
MODEL = init_model()
MODEL.eval(); MODEL.to("cuda")
logger.info("...Model loaded...")
ENCODER = load(open("baseline_encoder.pkl", "rb"))
logger.info("...Encoder loaded...")


def decode(preds):
    decoded = []
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    for i in range(preds.shape[0]):
        temp = []
        for k in preds[i, :]:
            k = k - 1
            if k == -1:
                temp.append("+")
            else:
                temp.append(ENCODER.inverse_transform([k])[0])
        decoded.append("".join(temp))
    return decoded

def base64toimg(data):
    pil_img = Image.open(BytesIO(base64.b64decode(data)))
    img = np.array(pil_img)
    return img

def make_prediction(img):
    img_tensor = get_valid_transforms()(image=img)["image"]
    img_tensor = img_tensor.unsqueeze(0).to("cuda")
    logger.info(img_tensor.shape)
    pred = MODEL(img_tensor)
    pred = pred.cpu().detach()
    text = decode(pred)
    return text

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
    img_base64 = body["base64"]
    img = base64toimg(img_base64)
    # logger.info(img.shape)
    text = make_prediction(img)

    logger.info(text)
    resp = {
        "resp": text
    }
    return jsonify(resp), 200

