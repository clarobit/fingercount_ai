# .h5 모델을 TFLite 포맷으로 변환하는 코드
# FP32 및 INT8 포맷 변환 지원

import tensorflow as tf
import numpy as np
from glob import glob
import cv2
import os

MODEL_PATH = "models/finger_count_cnn.h5"
DATA_DIR = "../data"
IMG_SIZE = 96

# Calibration 샘플 개수 (INT8 보정용)
CALIBRATION_SAMPLES = 100


def representative_dataset():
    files = glob(os.path.join(DATA_DIR, "*.png"))
    np.random.shuffle(files)
    sample_files = files[:CALIBRATION_SAMPLES]

    for file in sample_files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        yield [img.astype(np.float32)]


def convert_tflite():
    model = tf.keras.models.load_model(MODEL_PATH)

    # FP32 TFLite
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_fp32 = converter_fp32.convert()
    with open("models/finger_fp32.tflite", "wb") as f:
        f.write(tflite_model_fp32)
    print("Saved → models/finger_fp32.tflite")

    # INT8 TFLite
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8

    tflite_model_int8 = converter_int8.convert()
    with open("models/finger_int8.tflite", "wb") as f:
        f.write(tflite_model_int8)
    print("Saved → models/finger_int8.tflite")


if __name__ == "__main__":
    convert_tflite()
