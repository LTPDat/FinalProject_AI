import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer
from GUI import Ui_MainWindow
import cv2
import numpy as np
from keras.models import load_model

model = load_model("face_mask_detection.h5")  # loads the trained model

results = {0: 'without mask', 1: 'mask'}  # sets the class labels
GR_dict = {0: (0, 0, 255), 1: (0, 255, 0)}  # sets the color of the rectangle

rect_size = 4 # sets the size of the frame to be resized to

# loads the Haar cascade classifier to detect faces
haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        super().setupUi(self)
        
        self.btn_OpenWebcam.clicked.connect(self.openWebcam)
        self.btn_CloseWebcam.clicked.connect(self.closeWebcam)

    def openWebcam(self):
        self.video_capture = cv2.VideoCapture(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rerect_size = cv2.resize(frame, (frame.shape[1] // rect_size, frame.shape[0] // rect_size))
            # detects faces in the resized frame using a Haar cascade classifier
            # "detectMultiScale" function returns a list of rectangles representing the faces.
            faces = haarcascade.detectMultiScale(rerect_size)
            # loop over the face detections and draw them on the frame
            for f in faces:
                # scales up the coordinates of the detected face from the resized frame to the original frame
                (x, y, w, h) = [v * rect_size for v in f]
                # extracts the detected face region from the original frame
                face_frame = frame[y:y + h, x:x + w]
                # resizes the face image to 224x224 pixels as required by the model
                rerect_sized = cv2.resize(face_frame, (224, 224))
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

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.lbl_Webcam.setPixmap(pixmap)

    def closeWebcam(self):
        try:
            self.video_capture.release()
            self.lbl_Webcam.clear()
        except:
            print("Webcam is not open")

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()