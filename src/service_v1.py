import bentoml
from ultralytics import YOLO
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import asyncio
import time


@bentoml.service(
    workers=1, # if set to 1, requests will not run in parallel. But if set to 2, then model is duplicated
    # resources={   # the 'resources' field only takes effect on BentoCloud
    #     'cpu': '2',
    #     'memory': '2G',
    # },
    traffic={
        "concurrency": 5, # if set to 1, requests are processed sequentially without overlap.
        'timeout': 100
    }
)
class ObjectDetector:


    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.n_calls = 0
        print('[INFO] Model loaded')


    # @bentoml.api(route="/custom/url/name")
    @bentoml.api
    async def predict(self, base64_img: str):

        self.n_calls += 1

        img = np.array(
            Image.open(
                BytesIO(
                    base64.b64decode(base64_img)
                )
            )
        )[:, :, ::-1] # RGB to BGR

        print(f'[INFO] Predicting {self.n_calls} at {time.time()}')
        for i in range(10):
            time.sleep(1)
            print(f'Processing {i}')

        results = self.model(img)[0].cpu().numpy()
        boxes = results.boxes
        xywhs = boxes.xywh.tolist()
        confs = boxes.conf.tolist()
        classes = boxes.cls.tolist()

        return {
            'xywhs': xywhs,
            'confs': confs,
            'classes': classes
        }


    @bentoml.api
    def get_base64_img(self, img_path):
        with open(img_path, "rb") as img_file:
            return {
                # rgb
                'base64_img': base64.b64encode(img_file.read()).decode("utf-8")
            }