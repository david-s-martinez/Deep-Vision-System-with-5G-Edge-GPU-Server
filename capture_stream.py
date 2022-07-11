import json
import requests
import cv2
import numpy as np
import time
import math
import cv2.aruco as aruco
from threading import Thread
import time
from tool.darknet2pytorch import Darknet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import torch
import argparse
import cv2
import math
import numpy as np
from plane_computation.plane_detection import PlaneDetection

ONLINE = 1

X = 0
Y = 1
ROTATION = 0
TRANSFORM = 1

marker_size = 2.91  # cm
HEIGHT_CUBE = 3
grid_w = 28.4
# grid_w = 35
grid_h = 12.6
font = cv2.FONT_HERSHEY_SIMPLEX

camera_matrix = np.loadtxt('camera_matrix.txt', delimiter=',')
camera_distortion = np.loadtxt('distortion.txt', delimiter=',')

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()
first_flag = True
use_cuda = True
url_video = 'http://10.41.0.4:8080/?action=stream'
url_detections = 'http://10.41.0.4:5000/detections'
# url_video = 'http://10.41.0.4:8080/?action=stream'
# url_detections = 'http://192.168.100.43:5000/detections'

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h
    #print("aspect",aspect)
    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    print(scaled_img.shape)
    return scaled_img

def point_inside_prlgm(object_x, object_y, points):
    center_up = [int(points[4][X] + points[5][X]) // 2, int(points[4][Y] + points[5][Y]) // 2]
    center_right = [int(points[5][X] + points[6][X]) // 2, int(points[5][Y] + points[6][Y]) // 2]
    center_bott = [int(points[6][X] + points[7][X]) // 2, int(points[6][Y] + points[7][Y]) // 2]
    center_left = [int(points[7][X] + points[4][X]) // 2, int(points[7][Y] + points[4][Y]) // 2]
    center = [int(center_up[X] + center_bott[X]) // 2, int(center_up[Y] + center_bott[Y]) // 2]

    box0 = [center, center_left, points[4]]
    box1 = [center_right, center, center_up]
    box2 = [points[6], center_bott, center]
    box3 = [center_bott, points[7], center_left]
    i = 0
    for poly in [box0, box1, box2, box3]:
        xb = poly[0][0] - poly[1][0]
        yb = poly[0][1] - poly[1][1]
        xc = poly[2][0] - poly[1][0]
        yc = poly[2][1] - poly[1][1]
        xp = object_x - poly[1][0]
        yp = object_y - poly[1][1]
        d = xb * yc - yb * xc
        if d != 0:
            oned = 1.0 / d
            bb = (xp * yc - xc * yp) * oned
            cc = (xb * yp - xp * yb) * oned
            inside = (bb >= 0) & (cc >= 0) & (bb <= 1) & (cc <= 1)
            if inside:
                return i
        i += 1
    return None

def get_color(c, x, max_val):
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r * 255)

def plot_boxes(img, boxes,image_point_dict, index_of_marker,homog,class_names=None, color=None):
    img = np.copy(img)

    centroids = []
    rectangles = []
    labels = []
    confidences = []
    index_marker = []
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        
        bbox_thick = int(0.6 * (height + width) / 600)
        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            # print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            msg = str(class_names[cls_id])+" "+str(round(cls_conf,3))
            t_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
            c1, c2 = (x1,y1), (x2, y2)
            centroid = ((x1+x2)//2,(y1+y2)//2)
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        
            cv2.rectangle(img, (x1,y1), c3, rgb, -1)
            img = cv2.putText(img, msg, (c1[0], (c3[1])+15), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,0), bbox_thick//2,lineType=cv2.LINE_AA)
            if pd.box_verts_update:
                    image_point=[
                        image_point_dict[corners['tl']][0],
                        image_point_dict[corners['tr']][0],
                        image_point_dict[corners['br']][0],
                        image_point_dict[corners['bl']][0],
                        image_point_dict[corners['tl']][1],
                        image_point_dict[corners['tr']][1],
                        image_point_dict[corners['br']][1],
                        image_point_dict[corners['bl']][1]
                    ]
                    index_of_marker = point_inside_prlgm(centroid[0],centroid[1], image_point)

            # y=y-80
            centroids.append([x1, y1])
            rectangles.append([x2, y2])
            labels.append(int(cls_id))
            confidences.append(int(cls_conf * 100))
            index_marker.append(index_of_marker)
        if homog is not None:
                new_centroid = np.append(centroid,1)
                world_centroid = homog.dot(new_centroid)
                world_centroid = world_centroid[0], world_centroid[1]
                pos_str_x = str('x:'+str(round(world_centroid[0]/10,2)))
                pos_str_y = str('y:'+str(round(world_centroid[1]/10,2)))
                cv2.putText(img, 
                            pos_str_x, (centroid[0]-40,centroid[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1)
                cv2.putText(img, 
                            pos_str_y, (centroid[0]-40,centroid[1]+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, bbox_thick)
    data = {
        'centroids': centroids,
        'rectangles': rectangles,
        'labels': labels,
        'confidences': confidences,
        'markers': index_marker
    }
    return img, data

frame = None
frame2 = None
warp = None

def cam_reader():
    global frame
    global frame2
    global warp
    cap = cv2.VideoCapture(url_video)
    # cap = cv2.VideoCapture(2)
    # cap.set(3, 640)
    # cap.set(4, 480)
    while True:
        try:
            ret, frame = cap.read()
            # frame = resizeAndPad(frame, (416, 416), 0)
            height, width, channels = frame.shape
            
            print(frame.shape)
        except:
            print("CAMERA COULD NOT BE OPEN")
            break

        if(frame2 is None):
            frame2 = frame
        # cv2.imshow('frame', frame2)
        cv2.imshow('frame', cv2.resize(frame2, (1380,1020)))
        if warp is not None:
            cv2.imshow('warp', warp)

        key = cv2.waitKey(1)
        if key == 27:
            break


t1 = Thread(target=cam_reader, args=())
t1.start()

time.sleep(1)
calib_path = ""
corners = {
    'tl' :'0',
    'tr' :'1',
    'br' :'2',
    'bl' :'3'
    }

index_of_marker = -1
first_tag = True

pd = PlaneDetection(calib_path, corners)

m = Darknet('config.cfg')
m.print_network()
m.load_weights('yolo.weights')

if use_cuda:
    m.cuda()

class_names = ['DISK','TIRE','WHEEL']

while True:
    start_time = time.time()
    '''
    *********************** ARUCO  ***********************
    '''
    pd.detect_tags_3D(frame)
    image_point_dict = pd.box_verts_update
    # print(image_point_dict)
    
    
    homography = pd.compute_homog(w_updated_pts=True)
    warp = pd.compute_perspective_trans(frame, w_updated_pts=True)

    start_time2 = time.time()
    '''
    *********************** AI PART ***********************
    '''
    # frame = resizeAndPad(frame, (416, 416), 0)
    boxes = do_detect(m, frame, 0.5, 0.6, use_cuda)
    frame, data = plot_boxes(frame, boxes[0], image_point_dict, index_of_marker, homography ,class_names=class_names)
    frame2 = frame
    print(data)
    try:
        server_return = requests.post(url_detections, json=data)
        print('[INFO]: Detections posted.')
    except:
        break

    print("FPS:" + str(1.0 / (time.time() - start_time)))

