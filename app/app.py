import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

import cv2
import numpy as np

from ffpyplayer.player import MediaPlayer
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QWidget, QGroupBox, QFrame, QApplication, QLabel, QVBoxLayout, QGridLayout, QHBoxLayout, QMainWindow, QMenu, QAction, QFileDialog
import sys

cv2.ocl.setUseOpenCL(False)

class EmotionDetectionModel(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_prediction_signal = pyqtSignal(str)
    prediction_buffer_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',\
                              input_shape=(48,48,1)))

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        self.file_name = ""
        self.is_video_feed = False
        self.is_camera_feed = False

        self.load_model()

        self.opencv_init()

    def opencv_init(self):
        cv2.ocl.setUseOpenCL(False)

    def load_model(self):
        self.model.load_weights('model.h5')

    def model_run(self, frame):
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray,\
                                                                   (48, 48)), -1), 0)
            prediction = self.model.predict(cropped_img)
            self.prediction_buffer_signal_to_Qt(prediction)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60),\
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            self.prediction_signal_to_Qt(emotion_dict[maxindex])

    def camera_feed(self):
        cap = cv2.VideoCapture(0)

        while self.is_camera_feed:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()

            self.model_run(frame)
            self.video_signal_to_Qt(frame)

            if not ret:
                break

    def video_feed(self):
        video = cv2.VideoCapture(self.file_name)
        player = MediaPlayer(self.file_name)


        while self.is_video_feed:
            ret, video_frame = video.read()

            self.model_run(video_frame)
            self.video_signal_to_Qt(video_frame)

            if not ret:
                break

    def video_signal_to_Qt(self, frame):
        self.change_pixmap_signal.emit(frame)

    def prediction_signal_to_Qt(self, string):
        self.change_prediction_signal.emit(string)

    def prediction_buffer_signal_to_Qt(self, array):
        self.prediction_buffer_signal.emit(array)

    def run(self):
            self.camera_feed()
            self.video_feed()

class Widget(QMainWindow):
    def __init__(self):
        super().__init__() 

        self.display_width = 500
        self.display_height = 500

        self.image_label = QLabel()
        # self.image_label.resize(300, 300)

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.image_label)

        self.prediction_label = QLabel()
        self.prediction_label.setFrameStyle(QFrame.Panel)
        self.prediction_label.setText("Open Feed from Web Camera or Local Files \n Files-> ")
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 20pt;")

        self.prediction_buffer_headings = [QLabel("Angry"), QLabel("Disgusted"),\
                                           QLabel("Fearful"), QLabel("Happy"),QLabel("Neutral"),\
                                           QLabel("Sad"), QLabel("Suprised")]

        for i in range (7):
            self.prediction_buffer_headings[i].setStyleSheet("font-size: 16pt;")

        self.prediction_buffer_labels = [QLabel("0") for i in range(7)]

        prediction_layout = QGridLayout()

        outerprediction_layout = QVBoxLayout()

        inner_prediction_label_layouts = [QHBoxLayout() for i in range(7)]
        outerprediction_layout.addWidget(self.prediction_label)
        for i in range(7):
            inner_prediction_label_layouts[i].addWidget(self.prediction_buffer_headings[i])
            inner_prediction_label_layouts[i].addWidget(self.prediction_buffer_labels[i])

        prediction_layout.addLayout(outerprediction_layout, 0, 0)
        for i in range(7):
            prediction_layout.addLayout(inner_prediction_label_layouts[i], i + 1, 0)


        group_box = QGroupBox("")
        group_box.setLayout(prediction_layout)
        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout, 0)
        main_layout.addWidget(group_box)

        widget = QWidget()
        self.setCentralWidget(widget)
        widget.setLayout(main_layout)

        self.setStyleSheet("background-color: lightblue;")
        self.show()
        self.setWindowTitle("Emotion Recognition through Face")

        self.menu_bar_init()
        self.ml_model = EmotionDetectionModel()
        self.ml_model.change_pixmap_signal.connect(self.show_frame)
        self.ml_model.change_prediction_signal.connect(self.show_prediction)
        self.ml_model.prediction_buffer_signal.connect(self.update_prediction_buffer)


    def menu_bar_init(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        cam_feed_action = QAction('Web Cam', self)
        cam_feed_action.triggered.connect(self.video_from_webcam)

        video_feed_action = QAction('From Files', self)
        video_feed_action.triggered.connect(self.video_from_files)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.exit)

        file_menu.addAction(cam_feed_action)
        file_menu.addAction(video_feed_action)
        file_menu.addAction(exit_action)


    def video_from_webcam(self):
        self.ml_model.is_video_feed = False
        self.ml_model.is_camera_feed = True

        self.ml_model.start()

    def video_from_files(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        fileName = QFileDialog.getOpenFileName(self, "Open Files", "", "All Files (*);;", options=options)

        if fileName: 
            self.ml_model.file_name = fileName[0];
            self.ml_model.is_video_feed = True
            self.ml_model.is_camera_feed = False
            self.ml_model.start()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h,\
                                            bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width,\
                                        self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    @pyqtSlot(np.ndarray)   
    def show_frame(self, frame):
        q_img = self.convert_cv_qt(frame)
        self.image_label.setPixmap(q_img)

    @pyqtSlot(str)   
    def show_prediction(self, prediction):
        self.prediction_label.setText(prediction)

    @pyqtSlot(np.ndarray)   
    def update_prediction_buffer(self, prediction_buffer):
        for i in range(7):
            self.prediction_buffer_labels[i].setText(str(prediction_buffer[0][i]))

    def exit(self):
        sys.exit()

if __name__ == "__main__":
    app = QApplication([])

    a = Widget()
    a.show()

    sys.exit(app.exec_())
