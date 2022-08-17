import json
import requests
import cv2
import numpy as np
import time
import math
import cv2.aruco as aruco
from threading import Thread
import time

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

url_video = 'http://10.41.0.4:8080/?action=stream'
url_detections = 'http://10.41.0.4:5000/detections'

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

    return scaled_img

def draw_pose(image, camera_matrix, camera_distortion, rvec, tvec, z_rot=1):
    world_points = np.array([
        25.3, 0, 0,
        0, 0, 0,
        0, 13, 0,
        0, 0, 1 * z_rot
    ]).reshape(-1, 1, 3) * 0.5 * marker_size

    img_points, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, camera_distortion)
    img_points = np.round(img_points).astype(int)
    img_points = [tuple(pt) for pt in img_points.reshape(-1, 2)]

    cv2.line(image, img_points[1], img_points[0], (0, 0, 255), 2)
    cv2.line(image, img_points[1], img_points[2], (0, 255, 0), 2)
    cv2.line(image, img_points[1], img_points[3], (255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'X', img_points[0], font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'Y', img_points[2], font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'Z', img_points[3], font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, str((0, 0)), (img_points[1][0] + 10, img_points[1][1] - 30), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def define_world_pts(iD):
    if iD == 0:

        world_points = np.array([
            0, 0, 0,
            grid_w, 0, 0,
            grid_w, -grid_h, 0,
            0, -grid_h, 0,

            0, 0, HEIGHT_CUBE,
            grid_w, 0, HEIGHT_CUBE,
            grid_w, -grid_h, HEIGHT_CUBE,
            0, -grid_h, HEIGHT_CUBE

        ]).reshape(-1, 1, 3)

    elif iD == 1:
        world_points = np.array([
            -grid_w, 0, 0,
            0, 0, 0,
            0, -grid_h, 0,
            -grid_w, -grid_h, 0,

            -grid_w, 0, HEIGHT_CUBE,
            0, 0, HEIGHT_CUBE,
            0, -grid_h, HEIGHT_CUBE,
            -grid_w, -grid_h, HEIGHT_CUBE

        ]).reshape(-1, 1, 3)

    elif iD == 2:
        world_points = np.array([
            -grid_w, grid_h, 0,
            -grid_w, 0, 0,
            0, 0, 0,
            0, grid_h, 0,

            -grid_w, grid_h, HEIGHT_CUBE,
            -grid_w, 0, HEIGHT_CUBE,
            0, 0, HEIGHT_CUBE,
            0, grid_h, HEIGHT_CUBE,
        ]).reshape(-1, 1, 3)

    elif iD == 3:
        world_points = np.array([
            0, grid_h, 0,
            grid_w, grid_h, 0,
            grid_w, 0, 0,
            0, 0, 0,

            0, grid_h, 3,
            grid_w, grid_h, 3,
            grid_w, 0, 3,
            0, 0, 3

        ]).reshape(-1, 1, 3)

    elif iD == 4:
        world_points = np.array([
            12.75, 7, 0,
            -12.75, 7, 0,
            -12.75, -7, 0,
            12.75, -7, 0,
            12.75, 7, 3,
            -12.75, 7, 3,
            -12.75, -7, 3,
            12.75, -7, 3,
        ]).reshape(-1, 1, 3)
    else:
        world_points = np.array([
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,

            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0

        ]).reshape(-1, 1, 3)

    return world_points * 0.5 * marker_size


def draw_grid_id(image, img_points, rvec, tvec):
    img_points, _ = cv2.projectPoints(img_points, rvec, tvec, camera_matrix, camera_distortion)
    img_points = np.round(img_points).astype(int)
    img_points = [tuple(pt) for pt in img_points.reshape(-1, 2)]

    cv2.line(image, img_points[0], img_points[1], (255, 0, 0), 2)
    cv2.line(image, img_points[1], img_points[2], (255, 0, 0), 2)
    cv2.line(image, img_points[2], img_points[3], (255, 0, 0), 2)
    cv2.line(image, img_points[3], img_points[0], (255, 0, 0), 2)
    cv2.line(image, img_points[4], img_points[5], (255, 0, 0), 2)
    cv2.line(image, img_points[5], img_points[6], (255, 0, 0), 2)
    cv2.line(image, img_points[6], img_points[7], (255, 0, 0), 2)
    cv2.line(image, img_points[7], img_points[4], (255, 0, 0), 2)
    cv2.line(image, img_points[0], img_points[4], (255, 0, 0), 2)
    cv2.line(image, img_points[1], img_points[5], (255, 0, 0), 2)
    cv2.line(image, img_points[2], img_points[6], (255, 0, 0), 2)
    cv2.line(image, img_points[3], img_points[7], (255, 0, 0), 2)


def count_centroids(corners, ids):
    centers = {}
    for i in range(len(ids)):
        for point in corners[i].tolist():
            # print(point)
            x0 = point[0][X]
            y0 = point[0][Y]
            x2 = point[2][X]
            y2 = point[2][Y]
            centers[ids[i][0]] = np.array([(x0 + x2) // 2, (y0 + y2) // 2])
    return centers


def draw_cube(image, points):
     
    cv2.line(image, points[0], points[1], (255, 0, 0), 2)
    cv2.line(image, points[1], points[2], (255, 0, 0), 2)
    cv2.line(image, points[2], points[3], (255, 0, 0), 2)
    cv2.line(image, points[3], points[0], (255, 0, 0), 2)

    cv2.line(image, points[4], points[5], (255, 0, 0), 2)
    cv2.line(image, points[5], points[6], (255, 0, 0), 2)
    cv2.line(image, points[6], points[7], (255, 0, 0), 2)
    cv2.line(image, points[7], points[4], (255, 0, 0), 2)

    cv2.line(image, points[0], points[4], (255, 0, 0), 2)
    cv2.line(image, points[1], points[5], (255, 0, 0), 2)
    cv2.line(image, points[2], points[6], (255, 0, 0), 2)
    cv2.line(image, points[3], points[7], (255, 0, 0), 2)


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

frame = None
frame2 = None
def cam_reader():
    global frame
    global frame2
    cap = cv2.VideoCapture(url_video)
    while True:
        try:
            ret, frame = cap.read()
            height, width, channels = frame.shape
        except:
            print("CAMERA COULD NOT BE OPEN")
            break

        if(frame2 is None):
            frame2 = frame
        cv2.imshow('frame', frame2)
        key = cv2.waitKey(1)


t1 = Thread(target=cam_reader, args=())
t1.start()

time.sleep(1)


net = cv2.dnn.readNet("yolo.weights", "config.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

index_of_marker = -1
first_tag = True


while True:
    start_time = time.time()
    '''
    *********************** ARUCO  ***********************
    '''
    all_coordinates = []
    all_tags = {}
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if np.all(markerIds) is not None:
        centroids = count_centroids(markerCorners, markerIds)

        poses = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_size, camera_matrix, camera_distortion)
        rot_vecs, tran_vecs = poses[0], poses[1]

        defined_positions = [0, 0, 0, 0, 0, 0, 0, 0]
        image_point = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        world_points = np.array([
            0, 0, 0,
            0, 0, 3,
        ]).reshape(-1, 1, 3) * 0.5 * marker_size
        i = 0
        for tag_id in markerIds:
             if tag_id[0] not in [0, 1, 2, 3]:
                 #print("CONTINUE")
                 continue
             all_tags[tag_id[0]] = [rot_vecs[i][0], tran_vecs[i][0]]
             i += 1
        # PREPARE DATA

        all_tags = dict(sorted(all_tags.items()))

        # PREDEFINE THE CUBE
        try:
                rvec, tvec = all_tags[list(all_tags)[0]]
        except:
	        continue 
        defined_positions = define_world_pts(list(all_tags)[0])

        image_point, _ = cv2.projectPoints(defined_positions, rvec, tvec, camera_matrix, camera_distortion)
        image_point = np.round(image_point).astype(int)
        image_point = [tuple(pt) for pt in image_point.reshape(-1, 2)]

        # REBUILD CUBE POINTS:
        for tag, rot_trans in all_tags.items():
            rvec, tvec = rot_trans

            #cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)
            image_coordinates, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, camera_distortion)
            image_coordinates = np.round(image_coordinates).astype(int)
            image_coordinates = [tuple(pt) for pt in image_coordinates.reshape(-1, 2)]

            image_point[tag] = image_coordinates[0]
            image_point[tag + 4] = image_coordinates[1]

    cv2.aruco.drawDetectedMarkers(frame, markerCorners)
    #draw_cube(frame, image_point)
    frame2 = frame

    start_time2 = time.time()
    '''
    *********************** AI PART ***********************
    '''
    centroids = []
    rectangles = []
    labels = []
    confidences = []
    index_marker = []

    frame = resizeAndPad(frame, (416, 416), 0)

    # height, width, _ = frame.shape
    net.setInput(cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False))
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]

            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y = int(detection[0] * 640), int(detection[1] * 640)

                w, h = int(detection[2] * 640), int(detection[3] * 640)
                x, y = center_x - w // 2, center_y - h // 2

                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
		
                if np.all(markerIds is not None):
                    index_of_marker = point_inside_prlgm(center_x, center_y, image_point)

                skip = 0
                for i in all_coordinates:
                    if i[0] < center_x < i[0] + i[2] and i[1] < center_y < i[1] + i[3]:
                        skip = 1
                if skip:
                    continue
                else:
                    all_coordinates.append([x, y, w, h])
                y=y-80
                centroids.append([int(x), int(y)])
                rectangles.append([int(x + w), int(y + h)])
                labels.append(int(class_id))
                confidences.append(int(confidence * 100))
                index_marker.append(index_of_marker)
    data = {
        'centroids': centroids,
        'rectangles': rectangles,
        'labels': labels,
        'confidences': confidences,
        'markers': index_marker
    }
    print(data)
    try:
        server_return = requests.post(url_detections, json=data)
        print('[INFO]: Detections posted.')
    except:
        break

    print("FPS:" + str(1.0 / (time.time() - start_time)))

