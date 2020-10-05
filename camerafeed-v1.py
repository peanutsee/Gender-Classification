import cv2
import numpy as np
from model import Model

# Instantiate Model
model = Model()

def output_prediction(image):
    m = model.create_model()
    image = model.process(image)
    return model.predictor(m, image)

# Load Classifiers for Facial Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Convert Frame to GRAYSCALE
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Returns face in [x, y, w, h] relative to top left corner
    face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

    #define gender
    gender = 'female'

    for fx, fy, fw, fh in face:
        img = cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 1)
        #print(fx, fy, fw, fh)
        crop_img = img[fy:fy + fh, fx:fx + fw]
        #print(type(crop_img))
        cv2.imshow('crop', crop_img)

        #out = output_prediction(crop_img)
        f, m = output_prediction(crop_img)
        print(f, m)
        #if out == 1:
            #gender = 'male'

        cv2.putText(frame, 'Female {:.2f}% Male {:.2f}%'.format(f, m), (fx + 10, fy - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1) # Face Text

    if ret:
        cv2.imshow('Video Capture', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()