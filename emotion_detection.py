import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

class DataLoader: # 이미지 파일을 읽어와서 모델 학습용 데이터로 준비
    def __init__(self, emotions):
        self.emotions = emotions

    def load(self, train_dir, test_dir):
        def _load(folder):
            images, labels = [], []
            for idx, emo in enumerate(self.emotions):
                path = os.path.join(folder, emo)
                if not os.path.isdir(path):
                    continue
                for fname in os.listdir(path):
                    img = cv2.imread(os.path.join(path, fname), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    images.append(cv2.resize(img, (48, 48)))  # 크기를 48×48로 고정
                    labels.append(idx)
            return np.array(images), np.array(labels)

        X_train, y_train = _load(train_dir)
        X_test, y_test   = _load(test_dir)

        # 이미지 데이터를 0~1 범위로 정규화하고 모양 변환
        X_train = X_train.reshape(-1, 48, 48, 1) / 255.0
        X_test  = X_test.reshape(-1, 48, 48, 1) / 255.0

        # 정답 레이블을 원-핫 인코딩
        y_train = to_categorical(y_train, len(self.emotions))
        y_test  = to_categorical(y_test, len(self.emotions))

        return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    dl = DataLoader(emotions)
    try:
        X_train, X_test, y_train, y_test = dl.load('data/archive/train', 'data/archive/test')
        print("데이터 로딩 완료:", X_train.shape, y_train.shape)
    except Exception as e:
        print("데이터 로딩 테스트 중 에러 발생:", e)
