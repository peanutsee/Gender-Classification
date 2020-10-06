import cv2
import numpy as np
from model import Model

# Instantiate Model
model = Model()


def output_prediction(image):
    mod = model.create_model()
    image = model.process(image)
    return model.predictor(mod, image)


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
    face = face_cascade.detectMultiScale(frame_gray,
                                         scaleFactor=1.3,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    # Define gender
    gender = 'female'

    for fx, fy, fw, fh in face:
        face_frame = frame[fy-10:fy+fh+15, fx:fx+fw]
        crop_img = cv2.resize(face_frame, (128, 128))
        cv2.imshow('crop', crop_img)

        img = cv2.rectangle(frame, (fx, fy),
                            (fx + fw, fy + fh),
                            (255, 0, 0), 1)


        f, m = output_prediction(crop_img)
        print(f, m)

        cv2.putText(frame,
                    'Female {:.2f}% Male {:.2f}%'.format(f * 100, m * 100), (fx + 10, fy - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)  # Face Text
    if ret:
        cv2.imshow('Video Capture', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
