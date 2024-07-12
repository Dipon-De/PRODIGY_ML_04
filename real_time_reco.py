import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

model = load_model('hand_gesture_model.h5')

labels = ['Palm', 'l', 'Fist', 'Fist_moved', 'Thumb', 'Index', 'Ok', 'Palm_moved', 'C', 'Down']

cap = cv2.VideoCapture(0)


def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape
    x1, y1, x2, y2 = width // 2 - 112, height // 2 - 112, width // 2 + 112, height // 2 + 112
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    roi = frame[y1:y2, x1:x2]

    roi_processed = preprocess_image(roi)

    prediction = model.predict(roi_processed)
    predicted_label = labels[np.argmax(prediction)]

    cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
