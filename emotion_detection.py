import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

class ModelBuilder: # Hyperband 튜너에서 사용할 CNN 모델을 생성
    def build(hp):
        model = Sequential()

        # 3개의 합성곱 레이어 구성 (각 레이어에 하이퍼파라미터 적용)
        for i, filters in enumerate([
            hp.Choice('conv1_filters', [32, 64]),
            hp.Choice('conv2_filters', [64, 128]),
            hp.Choice('conv3_filters', [128, 256])
        ]):
            if i == 0:
                model.add(Conv2D(filters, (3, 3), activation='relu', input_shape=(48, 48, 1),
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     hp.Choice('l2', [1e-4, 1e-3])
                                 )))
            else:
                model.add(Conv2D(filters, (3, 3), activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     hp.Choice('l2', [1e-4, 1e-3])
                                 )))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(2, 2))

        model.add(Flatten())
        model.add(Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu'))
        model.add(Dropout(hp.Choice('dropout_rate', [0.2, 0.3, 0.5])))
        # 여기서는 감정 개수를 직접 지정 (7개)
        model.add(Dense(7, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        )
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

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

    class DummyHP:
        def Choice(self, name, values):
            return values[0]
        def Int(self, name, min_value, max_value, step):
            return min_value

    dummy_hp = DummyHP()
    model = ModelBuilder.build(dummy_hp)
    model.summary()
    
