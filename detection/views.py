from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.exceptions import APIException

import os
import time
import requests
from PIL import Image
import tensorflow as tf
import numpy as np
from io import BytesIO

class URLInvalid(APIException):
    status_code = 400
    default_detail = 'url이 유효하지 않습니다.'
    default_code = 'URLInvalid'

def load_model(dir):
    PATH_TO_SAVED_MODEL = os.path.join(dir, "saved_model")

    print('Loading model...', end='')
    start_time = time.time()
    detection_model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('The model took {} ms to load'.format(elapsed_time))

    return detection_model

detection_model = load_model(settings.DOG_FACE_DETECTION_MODEL)


def get_image_from_url(url):
    try:
        bytes_image = requests.get(url)
        image = Image.open(BytesIO(bytes_image.content))
        return np.array(image)
    except Exception:
        raise URLInvalid()


def get_detections(np_image):
    input_tensor = tf.convert_to_tensor(np_image)
    input_tensor = input_tensor[tf.newaxis, ...]
    input_tensor = input_tensor[:, :, :, :3]

    start_time = time.time()

    detections = detection_model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('The model took {} ms to predict'.format(elapsed_time))
    return detections


def get_b_boxes(detections):
    b_boxes = []
    for i in range(detections['num_detections']):
        if detections['detection_scores'][i] < settings.MIN_SCORE_THRESHOLD:
            break
        b_box_class = settings.CLASSES[detections['detection_classes'][i]]
        b_box_coordinate = detections['detection_boxes'][i].tolist()

        b_boxes.append({'class': b_box_class, 'coordinate': b_box_coordinate})
    return b_boxes


@api_view(["GET"])
def detection(request):
    url = request.GET.get('url')
    np_image = get_image_from_url(url)
    detections = get_detections(np_image)
    b_boxes = get_b_boxes(detections)

    return JsonResponse({'b_boxes': b_boxes})