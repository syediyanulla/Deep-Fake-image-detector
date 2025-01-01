import os
from image import image_pred
from PIL import Image
import streamlit as st
import traceback
import sys

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename, accepted_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in accepted_extensions

def process_image(image, model, dataset, threshold):
    try:
        Image.open(image).convert("RGB").save("uploads/check.jpg", "JPEG")
        output_string, pred = image_pred(
            image_path='uploads/check.jpg', model=model, dataset=dataset, threshold=threshold)
        return output_string, pred
    except Exception as e:
        return str(e), -1
    finally:
        os.remove("uploads/check.jpg")