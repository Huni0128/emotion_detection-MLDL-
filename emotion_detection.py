# 이 코드는 컴퓨터가 사람 얼굴 표정을 인식하도록 학습시키고, 웹캠 화면에 실시간으로 표정을 표시합니다.
import os
import cv2
import numpy as np
import tensorflow as tf 
import mediapipe as mp
import keras_tuner as kt                   # 모델 최적화(하이퍼파라미터 탐색) 도구
import tensorflow_model_optimization as tfmot  # 모델 경량화(Pruning 등)

from tensorflow.keras import mixed_precision # GPU 성능 향상을 위한 설정
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from collections import deque, Counter     # 최근 예측 기록 저장 및 가장 많이 나온 예측 계산

# GPU 연산을 더 빠르게 하기 위해 숫자를 절반(16비트)로 줄여 계산하도록 설정
mixed_precision.set_global_policy('mixed_float16')

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
                    images.append(cv2.resize(img, (48,48)))  # 크기를 48×48로 고정
                    labels.append(idx)      # 정답 레이블 저장
            return np.array(images), np.array(labels)

        X_train, y_train = _load(train_dir)
        X_test, y_test   = _load(test_dir)

        # 이미지 데이터를 0~1 범위로 정규화하고 모양 변환
        X_train = X_train.reshape(-1,48,48,1)/255.0
        X_test  = X_test.reshape(-1,48,48,1)/255.0

        # 정답 라벨을 '원-핫 인코딩' 형식으로 변환
        y_train = to_categorical(y_train, len(self.emotions))
        y_test  = to_categorical(y_test, len(self.emotions))

        return X_train, X_test, y_train, y_test

class ModelBuilder:
    # Hyperband 튜너가 사용할 CNN 모델을 생성
    @staticmethod
    def build(hp):
        model = Sequential()

        # 합성곱(Conv) 레이어 3개를 차례대로 쌓음
        for i, filters in enumerate([
            hp.Choice('conv1_filters', [32,64]),
            hp.Choice('conv2_filters', [64,128]),
            hp.Choice('conv3_filters', [128,256])
        ]):
            if i == 0:
                model.add(Conv2D(filters, (3,3), activation='relu', input_shape=(48,48,1),
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     hp.Choice('l2', [1e-4,1e-3])
                                 )))
            else:
                model.add(Conv2D(filters, (3,3), activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     hp.Choice('l2', [1e-4,1e-3])
                                 )))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(2,2))

        model.add(Flatten())  # 2D → 1D 변환
        model.add(Dense(hp.Int('dense_units', 64,256,step=64), activation='relu'))
        model.add(Dropout(hp.Choice('dropout_rate',[0.2,0.3,0.5])))
        model.add(Dense(len(emotions), activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(
            hp.Choice('learning_rate',[1e-2,1e-3,1e-4])
        )
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

class HyperbandTuner: # 자동으로 최적 하이퍼파라미터를 찾아주는 클래스
    def __init__(self):
        self.tuner = kt.Hyperband(
            ModelBuilder.build,
            objective='val_accuracy',
            max_epochs=30,
            factor=3,
            directory='kt_dir',
            project_name='emotion_tuning'
        )

    def search(self, X_train, y_train, X_test, y_test):
        prune_cb = tfmot.sparsity.keras.UpdatePruningStep()  # 모델 경량화 업데이트
        early = tf.keras.callbacks.EarlyStopping('val_accuracy', patience=5, restore_best_weights=True)

        # 튜닝 실행: 최적 모델 학습
        self.tuner.search(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            callbacks=[prune_cb, early]
        )

        best_model = self.tuner.get_best_models()[0]            # 최적 모델 반환
        best_hp    = self.tuner.get_best_hyperparameters()[0]   # 최적 하이퍼파라미터 반환
        return best_model, best_hp

class Distiller:
    # 큰(Teacher) 모델 → 작은(Student) 모델로 지식 전이
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student

    def distill(self, X_train, y_train, X_test, y_test):
        self.student.compile(
            optimizer='adam',
            loss=tf.keras.losses.KLDivergence(),  # Teacher 출력과 비교
            metrics=['accuracy']
        )
        self.student.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)
        return self.student

class Exporter:
    # 학습된 모델을 모바일/임베디드용 TFLite 파일로 변환
    @staticmethod
    def to_tflite(model, path):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        open(path, 'wb').write(converter.convert())

class EmotionRecognizer:
    # 웹캠으로 실시간 얼굴 감정 인식 + 예측 스무딩
    def __init__(self, model, emotions):
        self.model = model
        self.emotions = emotions
        self.history = deque(maxlen=10)  # 최근 10개 예측 저장

    def recognize(self):
        cap = cv2.VideoCapture(0)  # 웹캠 열기
        detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                for d in results.detections:
                    bbox = d.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w_box, h_box = (int(bbox.xmin*w), int(bbox.ymin*h),
                                          int(bbox.width*w), int(bbox.height*h))
                    
                    face_region = frame[y:y+h_box, x:x+w_box]
                    if face_region.size == 0:
                        continue  # 얼굴 영역이 비었으면 넘어감
                    face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

                    face = cv2.cvtColor(frame[y:y+h_box, x:x+w_box], cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (48,48)) / 255.0
                    preds = self.model.predict(face.reshape(1,48,48,1))
                    idx = np.argmax(preds)

                    self.history.append(idx)
                    emotion = self.emotions[Counter(self.history).most_common(1)[0][0]]

                    cv2.rectangle(frame, (x,y), (x+w_box, y+h_box), (0,255,0), 2)
                    cv2.putText(frame, emotion, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    # 1) 데이터 불러오기
    dl = DataLoader(emotions)
    X_train, X_test, y_train, y_test = dl.load('data/archive/train', 'data/archive/test')
    print("데이터 로딩 완료:", X_train.shape)

    # 2) Hyperband를 사용하여 최적의 Teacher 모델 학습
    tuner = HyperbandTuner()
    teacher, best_hp = tuner.search(X_train, y_train, X_test, y_test)
    print("최적 하이퍼파라미터:", best_hp.values)

    # 3) 작은 Student 모델 생성 및 Knowledge Distillation 적용
    student = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(emotions), activation='softmax')
    ])
    student = Distiller(teacher, student).distill(X_train, y_train, X_test, y_test)
    print("Knowledge Distillation 완료.")

    # 4) TFLite 파일로 모델 저장 (모바일/임베디드 배포용)
    Exporter.to_tflite(student, 'emotion_student_quant.tflite')

    # 5) 실시간 감정 인식 실행
    app = EmotionRecognizer(student, emotions)
    print("실시간 감정 인식 실행 중. 종료하려면 'q'를 누르세요.")
    app.recognize()

    
