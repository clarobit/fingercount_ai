# data를 불ㄹ러서 .h5 모델로 학습시키는 코드
# .h5 모델은 fp32 포맷이 기본이며, 이후 TFLite 변환으로 fp32, int8 등으로 변환 가능

import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

DATA_DIR = "../data"
IMG_SIZE = 96
EPOCHS = 50
BATCH = 32
NUM_CLASSES = 6


def load_dataset():
    images, labels = [], []
    files = glob(os.path.join(DATA_DIR, "*.png"))

    for file in files:
        label = int(os.path.basename(file).split("_")[1].split(".")[0])
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        images.append(img)
        labels.append(label)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    return train_test_split(
        images, labels, test_size=0.2, shuffle=True, random_state=42
    )


def build_model():
    # Sequential: 순서대로 층을 쌓는 방식 = 처음연산 -> 중간연산 -> 출력
    model = Sequential(
        [
            # relu: 음수를 0으로 바꾸고 양수는 그대로 통과시키는 활성화 함수
            # 1) 특징 추출
            # 입력: 이미지 96x96, 흑백(=1채널, rgb였다면 3채널) = 96x96
            # 출력: 96x96 이미지에 3x3필터를 16개 곱한 16채널 특징맵 = 96x96 이미지에 16종류의 필터를 적용한 96*96이미지 16장 = 96x96x16
            #
            # 16,(3,3): 크기가 3x3인 필터 16개를 사용하여 16개 특징맵을 가진 Conv2d 층 생성
            # 16: 16필터를 적용 = 각각 필터는 3x3 크기 = 필터는 가로를 강조, 세로를 강조 등으로 되어 있음
            # ex) 세로 강조 필터, 이 필터를 이미지에 적용하면 세로선이 강조된 이미지가 나옴
            #  0  1  0
            #  0  1  0
            #  0  1  0
            Conv2D(16, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            #
            # 2) 데이터 압축
            # 입력: 96x96 이미지 16장 = 96x96x16
            # 출력: 48x48 이미지 16장 = 48x48x16
            #
            # 가로세로 2*2 필터에서 최대값 추출
            # = 가장 특징이 큰 값만 추출
            # = 필터로 특징을 생성한 후 잡음 제거 및 압축
            MaxPool2D(2, 2),
            #
            # 3) 특징 추출 2
            # 입력 48×48×16  ⊗  필터 3×3×16  →  48×48×1  (특징맵 1장)
            # 입력: 48x48 16채널 특징맵을 가진 Conv2d 층 생성 = 48x48x16
            # 출력: 48x48*16 이미지에 3x3x16 필터를 32개 곱한 32채널 특징맵 = 48x48 이미지에 32종류의 필터를 적용한 48*48이미지 32장 = 48x48x32
            #
            # 32,(3,3): 크기가 3x3인 필터 16개를 사용하여 32개 특징맵을 가진 Conv2d 층 생성
            # 32: 32필터를 적용 = 각각 필터는 3x3x16(앞에서 채널 depth) 크기 = 필터는 가로를 강조, 세로를 강조 등으로 생겨 있음
            Conv2D(32, (3, 3), activation="relu"),
            MaxPool2D(2, 2),  # 48x48x32 -> 24x24x32
            Conv2D(64, (3, 3), activation="relu"),
            MaxPool2D(2, 2),  # 24x24x64 -> 12x12x64
            #
            # 4) 1차원 변환
            # Flatten: 다차원 배열을 1차원 배열로 변환해주는 층
            # 입력: 12x12 이미지 64장 = 12x12x64
            # 출력: 9216(12*12*64) 길이의 1차원 벡터
            #
            # 위의 과정 = 96x96 -> 96x96x16 -> 48x48x16 -> 48x48x32 -> 24x24x32 -> 24x24x64 -> 12x12x64
            # 앞에서 구해진 12x12x64를 1차원 벡터로 변환
            # 12x12x64 -> (9216,)
            Flatten(),
            #
            # 5) 최종판단1
            # Dense: 최종 판단
            # 입력: 9216 길이의 1차원 벡터
            # 출력: 64개의 뉴런을 완전 연결한 1차원 = 64 길이의 1차원 벡터
            #
            # 9216개의 입력 뉴런과 64개의 출력 뉴런을 완전 연결
            # 입력 뉴런 9216개 → 가중치 → 출력 뉴런 64개
            #
            # 수식 예시:
            # y1 = w1·x1 + w2·x2 + ... + w9216·x9216 + b
            # y2 = ...
            # ...
            # y64 = ...
            Dense(64, activation="relu"),
            #
            # 6) 과적합 방지
            # Dropout: 학습 시 특정 비율의 뉴런을 랜덤하게 비활성화하여 과적합을 방지하는 층
            # 학습중 20%의 뉴런을 랜덤하게 비활성화
            Dropout(0.2),
            #
            # 7) 최종판단2 = 클래스의 갯수만큼 출력 뉴런 생성 = 최종적으로 숫자 0~5 중 하나로 분류
            # 입력: 64 길이의 1차원 벡터
            # 출력: 6개의 뉴런을 완전 연결한 1차원 = 6 길이의 1차원 벡터 = 각 클래스(0~5)에 대한 확률값
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train():
    # 훈련 데이터 / 테스트 데이터로 나눔
    X_train, X_test, y_train, y_test = load_dataset()

    # 모델 생성
    model = build_model()

    # 실제로 학습하는 부분
    # validation_data는 학습 중 테스트 데이터를 함께 평가해서 과적합 여부 확인
    # history에는 **훈련 과정(loss, accuracy 등 변화 기록)**이 저장됨 = 에포크마다 변화 기록
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=BATCH,
        epochs=EPOCHS,
    )

    # 모델 저장
    model.save("models/finger_count_cnn.h5")
    print("Model saved → finger_count_cnn.h5")

    # 테스트 데이터로 최종 평가
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    return history, model


if __name__ == "__main__":
    train()
