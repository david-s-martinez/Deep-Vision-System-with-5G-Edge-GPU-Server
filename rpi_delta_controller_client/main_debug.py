#1

import numpy as np
import flask
from flask import Flask,render_template,Response,request
import cv2
# from picamera.array import PiRGBArray
# from picamera import PiCamera
# from camera import VideoCamera
import time
import threading
import os
import json
# import screeninfo
import threading
# os.system('export DISPLAY=:0')
time.sleep(5)

#camera = VideoCamera()
# camera = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream')
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

app = Flask(__name__)
colors = [(225,0,0),(0,225,0),(0,0,225)]
classes = ['DISK1','TIRE','WHEEL1']

def generate_frames(camera):
    # data_path = '/home/pi/AI_CV/data.json'
    data_path = 'data.json'
    while True:
        ret, frame_enc = camera.read()#get_frame()
        img = frame_enc.copy()#camera.frame
        
        with open(data_path,'r') as f:
            try:
                detections = json.load(f)
            except:
                detections = {"centroids": [],
                              "rectangles": [],
                              "labels": [],
                              "confidences": [],
                              "markers": []
                              }
            
        centroids = detections['centroids']
        rectangles = detections['rectangles']
        labels = detections['labels']
        confidences = detections['confidences']
        markers = detections['markers']

        
        if centroids:
            for i in range(len(centroids)):
                if int(rectangles[i][0]) / int(rectangles[i][1]) >= 2:
                    print("CANCEL THE DETECTION")
                    continue

                cv2.rectangle(img, (centroids[i][0], centroids[i][1]),
                             (rectangles[i][0], rectangles[i][1]), colors[labels[i]],2)
                #print("detection amount:",len(centroids))
                #print("sizes:", int(rectangles[i][0]),"/", int(rectangles[i][1]))
                #if int(rectangles[i][0]) / int(rectangles[i][1]) >= 2:
                #    print("CANCEL THE DETECTION")
                #    continue
                #print("sizes:", int(rectangles[i][0]),"/", int(rectangles[i][1]))

                if classes[labels[i]] == "DISK1":
                    place = 4
                    name = "DISC"
                elif classes[labels[i]] == "WHEEL1":
                    place = 2
                    name = "WHEEL"
                else:
                    place = 4
                    name = "TIRE"
                cv2.putText(img, str(name + str(confidences[i])+'%'),
                            (centroids[i][0], centroids[i][1]-10),
                            cv2.FONT_HERSHEY_PLAIN, 2, colors[labels[i]], 3)
                #cv2.putText(img,"#"+str(place)+str(name + str(confidences[i])+'%'),
                #           (centroids[i][0], centroids[i][1]-10),
                #           cv2.FONT_HERSHEY_PLAIN, 2, colors[labels[i]], 3)

                cv2.rectangle(img,(centroids[i][0],centroids[i][1]),
                              (rectangles[i][0],rectangles[i][1]),colors[labels[i]],2)
                #cv2.putText(img,"#"+str(markers[i])+str(classes[labels[i]] + str(confidences[i])+'%'),
                #            (centroids[i][0],centroids[i][1]+30),
                #            cv2.FONT_HERSHEY_PLAIN,2,(225,225,225),2)
                
        # print(img.shape)
        to_show = cv2.resize(img, (800, 480))
        # cv2.imshow('detection',img)
        cv2.imshow('detection',to_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            cv2.destroyAllWindows()
            camera.release()
            break
        
        #yield(b'--frame\r\n'
                   #b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n') #+ frame_enc +

#to get json detection data in same port
@app.route('/detections', methods = ['POST'])
def detections():
    try:
        request_data = request.get_json()
        with open('data.json','w') as f:
            json.dump(request_data,f)
        return Response(print(request_data))
    except:
        return Response(print('[INFO]:Failed to update data.json'))

window_name = "detection"

if __name__=="__main__":
    # screen = screeninfo.get_monitors()[0]
    # width, height = screen.width, screen.height
    # #width, height = 800, 480 
    # image = np.ones((height, width, 3), dtype=np.float32)
    ret, frame_test = camera.read()
    height, width, _ = frame_test.shape
    image = np.ones((height, width, 3), dtype=np.float32)
   
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, image)
    
    threading.Thread(target=lambda: app.run(host='0.0.0.0',port=5000,debug=False)).start()
    generate_frames(camera)
