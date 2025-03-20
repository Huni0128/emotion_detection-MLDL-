import os
import cv2
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import tensorflow_model_optimization as tfmot

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

class Exporter: # 학습된 모델을 모바일/임베디드용 TFLite 파일로 변환
    @staticmethod
    def to_tflite(model, path):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite 모델이 '{path}'에 저장되었습니다.")

class Distiller: # Teacher 모델의 지식을 Student 모델로 전이하는 지식 증류(knowledge distillation)
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student

    def distill(self, X_train, y_train, X_test, y_test):
        self.student.compile(
            optimizer='adam',
            loss=tf.keras.losses.KLDivergence(),  # Teacher 출력과의 차이를 최소화
            metrics=['accuracy']
        )
        self.student.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
        return self.student
    
class HyperbandTuner: # keras_tuner.Hyperband를 이용하여 최적의 하이퍼파라미터를 탐색
    def __init__(self):
        self.tuner = kt.Hyperband(
            ModelBuilder.build,
            objective='val_accuracy',
            max_epochs=3,  # 테스트용으로 에폭 수를 줄임
            factor=3,
            directory='kt_dir',
            project_name='emotion_tuning'
        )

    def search(self, X_train, y_train, X_test, y_test):
        prune_cb = tfmot.sparsity.keras.UpdatePruningStep()  # 모델 경량화 업데이트
        early = tf.keras.callbacks.EarlyStopping('val_accuracy', patience=2, restore_best_weights=True)

        # 튜닝 실행: 최적 모델을 찾기 위한 학습 진행
        self.tuner.search(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=3,
            callbacks=[prune_cb, early]
        )

        best_model = self.tuner.get_best_models()[0]         # 최적 모델
        best_hp = self.tuner.get_best_hyperparameters()[0]     # 최적 하이퍼파라미터
        return best_model, best_hp
    
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

    X_train = np.random.rand(20, 48, 48, 1)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 7, 20), 7)
    X_test = np.random.rand(5, 48, 48, 1)
    y_test = tf.keras.utils.to_categorical(np.random.randint(0, 7, 5), 7)

    tuner = HyperbandTuner()
    best_model, best_hp = tuner.search(X_train, y_train, X_test, y_test)
    print("최적 하이퍼파라미터:", best_hp.values)

    import numpy as np
    # Teacher: 간단 모델 (실제 튜닝된 모델 대신 더미 모델)
    teacher = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(),
        Flatten(),
        Dense(7, activation='softmax')
    ])
    teacher.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Student: 더 작은 모델
    student = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(),
        Flatten(),
        Dense(7, activation='softmax')
    ])

    # 더미 데이터 생성
    X_train = np.random.rand(20, 48, 48, 1)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 7, 20), 7)
    X_test = np.random.rand(5, 48, 48, 1)
    y_test = tf.keras.utils.to_categorical(np.random.randint(0, 7, 5), 7)

    distiller = Distiller(teacher, student)
    student = distiller.distill(X_train, y_train, X_test, y_test)
    print("Knowledge Distillation 테스트 완료.")

    model = Sequential([
        Flatten(input_shape=(48, 48, 1)),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 더미 데이터로 모델 학습 (테스트용)
    X = np.random.rand(10, 48, 48, 1)
    y = tf.keras.utils.to_categorical(np.random.randint(0, 7, 10), 7)
    model.fit(X, y, epochs=1)

    Exporter.to_tflite(model, 'test_emotion.tflite')
    
