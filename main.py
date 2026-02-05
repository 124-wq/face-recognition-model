import cv2 #computer vision 2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from deepface import DeepFace


video = cv2.VideoCapture(0) #0 is for default camera also VideoCapture is a class in cv2
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data = [] #making a python list to store information of face data
i = 0 #this counter will count number of faces captured


while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (200, 200))
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1

        result = DeepFace.analyze(resized_img, actions=['emotion'], enforce_detection=False)
        #extract dominant emotion
        emotion = result[0]['dominant_emotion']
        cv2.putText(frame, emotion, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        

        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        face_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()