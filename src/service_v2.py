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
    workers=1,
    traffic={
        'timeout': 100
    }
)
class ModelCore:

    def __init__(self):
        self.model = YOLO('yolov8n.pt').to('cuda')
        print('[INFO] Model loaded')
    

    @bentoml.api
    async def predict(self, img: np.ndarray):

        print(f'[INFO] Predicting {img.shape} at {time.time()}') # :((

        results = self.model(img)[0].cpu().numpy()
        boxes = results.boxes
        xywhs = boxes.xywh.tolist()
        confs = boxes.conf.tolist()
        classes = boxes.cls.tolist()

        for i in range(10):
            time.sleep(1)
            print(f'Sleeping {i}')

        return {
            'xywhs': xywhs,
            'confs': confs,
            'classes': classes
        }


@bentoml.service(
    workers=2, # if set to 1, then it'll be 10s slower if we call 2 requests at the same time
    traffic={
        'timeout': 100
    }
)
class ObjectDetector:

    
    core_service = bentoml.depends(ModelCore)


    def __init__(self):
        self.n_calls = 0


    @bentoml.api
    async def predict(self, base64_img: str):

        print(f'[INFO] Start {self.n_calls} at {time.time()}')

        self.n_calls += 1

        # preprocess
        for i in range(10):
            time.sleep(1)
            print(f'Processing step {i}')
        img = np.array(
            Image.open(
                BytesIO(
                    base64.b64decode(base64_img)
                )
            )
        )[:, :, ::-1] # RGB to BGR

        ret = await self.core_service.predict(img)

        print(f'[INFO] End {self.n_calls} at {time.time()}')

        return ret


    @bentoml.api
    def get_base64_img(self, img_path):
        with open(img_path, "rb") as img_file:
            return {
                # rgb
                'base64_img': base64.b64encode(img_file.read()).decode("utf-8")
            }

# play with both workers and concurrency to get insights