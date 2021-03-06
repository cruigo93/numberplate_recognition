import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
import requests
import json


def imgtobase64(img):
    pil_img = Image.fromarray(img)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def main():
    img_name = "../test_images/961UEB01.png"
    img = cv2.imread(img_name, 1)
    img_base64 = imgtobase64(img)
    data = {
        "base64": img_base64
    }
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post("http://localhost:9090/predict", data=json.dumps(data), headers=headers)
    print(r)



if __name__ == "__main__":
    main()