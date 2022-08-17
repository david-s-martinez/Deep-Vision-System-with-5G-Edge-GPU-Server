import cv2
from imutils.video.pivideostream import PiVideoStream
import imutils
import time
import numpy as np

class VideoCamera(object):
    def __init__(self):
        self.stream = PiVideoStream(rotation=180, resolution=(640, 480)).start()
        time.sleep(2.0)

    def __del__(self):
        self.stream.stop()

    def get_frame(self):
        self.frame = self.stream.read()
        ret, jpeg = cv2.imencode('.JPEG', self.frame)
        return jpeg.tobytes()
        
    def get_frame_raw(self):
        return self.frame
