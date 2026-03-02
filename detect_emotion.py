import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

model = load_model("emotion_model.hdf5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_prediction_time = 0
current_emotion = "Detecting..."

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Only update prediction every 1 second
    if time.time() - last_prediction_time > 1:
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48))
            face = face / 255.0
            face = np.reshape(face, (1,48,48,1))

            prediction = model.predict(face, verbose=0)
            current_emotion = emotion_labels[np.argmax(prediction)]

        last_prediction_time = time.time()

    # Draw rectangle & show last detected emotion
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, current_emotion, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()