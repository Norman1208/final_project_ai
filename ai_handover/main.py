import numpy as np
from torchvision import transforms
from PIL import Image
import requests
import cv2

import time
from ai_handover.initialization import init_classification_model, TheModelClass
from ai_handover.preprocessing import rn50_preprocess, preprocess_std
from inference import inference_model, inference_model_pt
from ai_handover.postprocessing import result_information, result_info_fashion_highest


def ai_pipeline(model, image64):
    # # preprocessing function

    transformed_img = preprocess_std(image64)

    inferenced_result = inference_model_pt(model=model,
                                           image_data=transformed_img)

    highest_result_name, highest_result_score = result_info_fashion_highest(
        results=inferenced_result)

    return highest_result_name, highest_result_score


def main():
    # from API
    image_base64 = requests.data["image64"]
    # converting from base64 to image cv
    # base64 processing to cv image
    # reconstruct image as an numpy array
    image64_decode = imread(io.BytesIO(base64.b64decode(image_base64)))

    # finally convert RGB image to BGR for opencv
    # and save result
    image_cv = cv2.cvtColor(image64_decode, cv2.COLOR_RGB2BGR)

    model = init_classification_model(model_class=TheModelClass)
    image_path = "./image.png"

    image_cv = cv2.imread(image_path)
    # reques dari api
    result_name, result_score = ai_pipeline(model, image_cv)

    print(result_name, result_score)

    # for API
    response = {
        "classification_name": result_name,
        "classification_score": result_score
    }
