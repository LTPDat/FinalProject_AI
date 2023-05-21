import cv2
import numpy as np
from keras.models import load_model

model = load_model("face_mask_detection.h5")  # loads the trained model

results = {0: 'without mask', 1: 'mask'}  # sets the class labels
GR_dict = {0: (0, 0, 255), 1: (0, 255, 0)}  # sets the color of the rectangle

rect_size = 4 # sets the size of the frame to be resized to

# loads the Haar cascade classifier to detect faces
haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0) # captures the video from the webcam
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1, 1)  # flip the frame
    # resizes the frame to a smaller size for faster face detection
    rerect_size = cv2.resize(frame, (frame.shape[1] // rect_size, frame.shape[0] // rect_size))
    # detects faces in the resized frame using a Haar cascade classifier
    # "detectMultiScale" function returns a list of rectangles representing the faces.
    faces = haarcascade.detectMultiScale(rerect_size)
    # loop over the face detections and draw them on the frame
    for f in faces:
        # scales up the coordinates of the detected face from the resized frame to the original frame
        (x, y, w, h) = [v * rect_size for v in f]
        # extracts the detected face region from the original frame
        face_img = frame[y:y + h, x:x + w]
        # resizes the face image to 224x224 pixels as required by the model
        rerect_sized = cv2.resize(face_img, (224, 224))
        # normalizes the pixel values of the face image to a range of 0-1
        normalized = rerect_sized / 255.0
        # reshapes the normalized face image
        reshaped = np.reshape(normalized, (1, 224, 224, 3))
        # stacks the reshaped face image into a vertical array
        reshaped = np.vstack([reshaped])
        # makes a prediction on the reshaped face image
        result = model.predict(reshaped)
        # selects the class label with the highest probability from the prediction result
        label = np.argmax(result, axis=1)[0]
        # get the confidence value of the prediction result
        confidence = result[0][label]
        # puts the class label and confidence value on the variable "label_text"
        label_text = f"{results[label]} ({confidence:.2f})"
        # draws a rectangle around the face detected
        cv2.rectangle(frame, (x, y), (x + w, y + h), GR_dict[label], 2)
        # draws a rectangle filled with the class label
        cv2.rectangle(frame, (x, y - 40), (x + w, y), GR_dict[label], -1)
        # writes the class label and confidence value on the rectangle
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Real time face mask detection', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
