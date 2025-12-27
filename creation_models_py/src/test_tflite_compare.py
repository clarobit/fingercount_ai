# .h5 모델과 TFLite 모델의 성능 비교 테스트 코드
# 동일한 테스트 데이터셋을 사용하여 FP32 및 INT8 TFLite 모델 비교

import os
import cv2
import numpy as np
from glob import glob
import random
import tensorflow as tf

DATA_DIR = "../data"
IMG_SIZE = 96
NUM_PER_CLASS = 20
NUM_CLASSES = 6

FP32_PATH = "models/finger_fp32.tflite"
INT8_PATH = "models/finger_int8.tflite"


def preprocess(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)


def extract_label(file):
    return int(os.path.basename(file).split("_")[1].split(".")[0])


def load_tflite(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    input_details = interpreter.get_input_details()[0]
    return interpreter, input_index, output_index, input_details


def evaluate_model(model_path):
    interpreter, input_idx, output_idx, input_details = load_tflite(model_path)

    # 입력 타입 판단
    is_int8 = (input_details['dtype'] == np.int8)
    if is_int8:
        scale, zero_point = input_details['quantization']

    files = glob(os.path.join(DATA_DIR, "*.png"))
    class_files = {i: [] for i in range(NUM_CLASSES)}
    for f in files:
        class_files[extract_label(f)].append(f)

    correct = {i: 0 for i in range(NUM_CLASSES)}
    tested = {i: 0 for i in range(NUM_CLASSES)}

    for c in range(NUM_CLASSES):
        random.shuffle(class_files[c])
        sample = class_files[c][:NUM_PER_CLASS]

        for file in sample:
            img = preprocess(file)  # float32 [0~1]

            if is_int8:
                # FP32 → INT8
                img = img / scale + zero_point
                img = img.astype(np.int8)

            interpreter.set_tensor(input_idx, img)
            interpreter.invoke()
            probs = interpreter.get_tensor(output_idx)[0]
            pred = np.argmax(probs)
            true = extract_label(file)

            tested[true] += 1
            if pred == true:
                correct[true] += 1

    total_correct = sum(correct.values())
    total_tested = sum(tested.values())
    total_acc = total_correct / total_tested * 100
    return correct, tested, total_acc


if __name__ == "__main__":
    print("\n===== FP32 Evaluation =====")
    fp32_correct, fp32_tested, fp32_acc = evaluate_model(FP32_PATH)
    print(f"Total Accuracy FP32: {fp32_acc:.2f}%")

    print("\n===== INT8 Evaluation =====")
    int8_correct, int8_tested, int8_acc = evaluate_model(INT8_PATH)
    print(f"Total Accuracy INT8: {int8_acc:.2f}%")

    print("\n===== Class Accuracy Comparison =====\n")
    for c in range(NUM_CLASSES):
        fp32 = fp32_correct[c] / fp32_tested[c] * 100
        int8 = int8_correct[c] / int8_tested[c] * 100
        print(f"Class {c} | FP32: {fp32:.2f}%   INT8: {int8:.2f}%")
