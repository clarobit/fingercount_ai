# .h5 모델을 로드 테스트 코드
# 각 클래스당 지정한 갯수만큼 균일하게 랜덤으로 샘플링하여 테스트


import os
import cv2
import random
import numpy as np
from glob import glob
from tensorflow.keras.models import load_model

DATA_DIR = "../data"
MODEL_PATH = "models/finger_count_cnn.h5"
IMG_SIZE = 96

# 클래스당 테스트할 이미지 수 (부족하면 자동으로 줄어듦)
NUM_PER_CLASS = 20   # ← 원하는 개수로 수정

NUM_CLASSES = 6  # 0~5 손가락


def preprocess(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def extract_label(filename):
    return int(os.path.basename(filename).split("_")[1].split(".")[0])


def test_uniform_per_class():
    print("\n===== Uniform Class Sampling Test =====\n")
    model = load_model(MODEL_PATH)
    files = glob(os.path.join(DATA_DIR, "*.png"))

    # 클래스별 이미지 분리
    class_files = {i: [] for i in range(NUM_CLASSES)}
    for file in files:
        lbl = extract_label(file)
        if lbl in class_files:
            class_files[lbl].append(file)

    total_correct = 0
    total_tested = 0
    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_tested = {i: 0 for i in range(NUM_CLASSES)}

    selected = []

    # 클래스별 균일 샘플링 + 부족할 경우 자동 보정
    for c in range(NUM_CLASSES):
        total_available = len(class_files[c])
        num_to_pick = min(NUM_PER_CLASS, total_available)

        if total_available == 0:
            print(f"[Warning] Class {c} has no images — skipped.\n")
            continue

        if num_to_pick < NUM_PER_CLASS:
            print(f"[Notice] Class {c}: requested {NUM_PER_CLASS} but only {total_available} available → testing {num_to_pick}\n")

        random.shuffle(class_files[c])
        selected.extend(class_files[c][:num_to_pick])

    # 테스트 루프
    for file in selected:
        true_label = extract_label(file)
        img = preprocess(file)

        probs = model.predict(img, verbose=0)[0]       # shape: (6,)
        pred = np.argmax(probs)

        # Top-2 확률 추출
        top2_idx = probs.argsort()[-2:][::-1]          # 확률 높은 2개
        top2_info = [(i, probs[i]) for i in top2_idx]  # (클래스, 확률)

        result = "O" if pred == true_label else "X"
        print(f"{result}  File: {os.path.basename(file)}  True = {true_label} / Pred = {pred}")

        # 확률 출력
        print("     Top2:", end=" ")
        for cls, p in top2_info:
            print(f"({cls}: {p:.2f})", end=" ")
        print()  # 줄바꿈

        class_tested[true_label] += 1
        total_tested += 1
        if pred == true_label:
            class_correct[true_label] += 1
            total_correct += 1

    # 결과 출력
    print("\n----------------------------------------")
    for c in range(NUM_CLASSES):
        if class_tested[c] == 0:
            print(f"Class {c} → No samples tested")
            continue
        acc = class_correct[c] / class_tested[c] * 100
        print(f"Class {c} → {class_correct[c]}/{class_tested[c]}  ({acc:.2f}%)")
    print("----------------------------------------")

    if total_tested > 0:
        total_acc = total_correct / total_tested * 100
        print(f"Total Tested : {total_tested}")
        print(f"Total Correct: {total_correct}")
        print(f"Total Accuracy: {total_acc:.2f}%")
    else:
        print("No images were tested.")
    print("----------------------------------------\n")


if __name__ == "__main__":
    test_uniform_per_class()
