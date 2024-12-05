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
import torch.nn
import torchvision
import argparse
import cv2
import math
import numpy as np
from conv_net_detect.test_model_Delta_v3 import *
from plane_detector.plane_computation.plane_detection import PlaneDetection
from multiprocessing import Process
from multiprocessing import Pipe
import torch.multiprocessing as mp

def get_color(c, x, max_val):
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r * 255)

def plot_boxes(img ,out_img, boxes,image_point_dict, index_of_marker,homog,corners,
    class_names=None, color=None, plane_dims=None):
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
        x_circle = int(box[6] * width)
        y_circle = int(box[7] * height)
        bbox_thick = int(0.6 * (height + width) / 600)
        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 6 and class_names:
            cls_conf = box[4]
            cls_id = round(box[5])
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            
            msg = str(class_names[int(cls_id)])+" "+str(round(cls_conf,3))
            t_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
            
            c1, c2 = (x1,y1), (x2, y2)
            centroid = ((x1+x2)//2,(y1+y2)//2)
            world_centroid = (x_circle*plane_dims['w']*10)/width,(y_circle*plane_dims['h']*10)/height
            w = int(abs(x2-x1) * 2.2)
            h = int(abs(y2-y1) * 2.2)

            inv_trans = np.linalg.pinv(homog)

            c = cv2.perspectiveTransform(np.float32([[[centroid[0], centroid[1]]]]), inv_trans)
            c_circle = cv2.perspectiveTransform(np.float32([[[x_circle, y_circle]]]), inv_trans)
            centroid = (int(c[0][0][0]),int(c[0][0][1]))
            centroid_circle = (int(c_circle[0][0][0]),int(c_circle[0][0][1]))
            x1 = centroid[0]-w//2
            y1 = centroid[1]-h//2
            x2 = w//2+centroid[0]
            y2 = h//2+centroid[1]
            c1, c2 = (x1,y1), (x2, y2)
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        
            cv2.rectangle(out_img, (x1,y1), c3, rgb, -1)
            out_img = cv2.putText(out_img, 
                                    msg, (c1[0], (c3[1])+15), 
                                    cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,0), 
                                    bbox_thick//2,lineType=cv2.LINE_AA)
            pos_str_x = str('x:'+str(round(world_centroid[0]/10,2)))
            pos_str_y = str('y:'+str(round(world_centroid[1]/10,2)))
            cv2.putText(out_img, 
                        pos_str_x, (centroid[0]-40,centroid[1]+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1)
            cv2.putText(out_img, 
                        pos_str_y, (centroid[0]-40,centroid[1]+40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1)
            cv2.circle(out_img, centroid_circle, 3, rgb)
            if cls_id ==0:
                cls_id = 1
            elif cls_id ==1:
                cls_id = 0
            centroids.append([x1, y1])
            rectangles.append([x2, y2])
            labels.append(int(cls_id))
            confidences.append(int(cls_conf * 100))
            index_marker.append(index_of_marker)
                
        out_img = cv2.rectangle(out_img, (x1, y1), (x2, y2), rgb, bbox_thick)
    data = {
        'centroids': centroids,
        'rectangles': rectangles,
        'labels': labels,
        'confidences': confidences,
        'markers': index_marker
    }
    return out_img, data

def cam_reader(cam_out_conn, cam_source):
    cap = cv2.VideoCapture(cam_source)
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            height, width, channels = frame.shape
            cam_out_conn.send(frame)
        except:
            print("CAMERA COULD NOT BE OPEN")
            break

def robot_perception(percept_in_conn, percept_out_conn, config, use_cuda = True):
    pd = PlaneDetection(config['vision'], 
                        config['plane']['corners'], 
                        marker_size = config['plane']['tag_size'], 
                        tag_scaling = config['plane']['tag_scaling'], 
                        box_z = config['plane']['z_tansl'],
                        tag_dict = config['plane']['tag_dict'])
    model = DetectionModel(number_classes=NUMBER_CLASSES, grid_size_width=GRID_SIZE_WIDTH, grid_size_height=GRID_SIZE_HEIGHT,chosen_model=2)
    model.load_state_dict(torch.load(config['neural_net']['model_config'], map_location=torch.device(0)))
    
    # model = torch.load(config['neural_net']['model_config'],map_location=torch.device(0))
    model.eval()
    model.cuda()
    disk_centroid_templates = [cv2.imread("conv_net_detect/disk_centroid_template_1.png"),
                                cv2.imread("conv_net_detect/disk_centroid_template_2.png"),
                                 cv2.imread("conv_net_detect/disk_centroid_template_3.png")]
    #disk_centroid_templates = [cv2.imread("conv_net_detect/disk_centroid_template_1.png"),cv2.imread("conv_net_detect/disk_centroid_template_3.png")]
    class_names = ['TIRE','DISK','WHEEL']
    frame = None
    warp = None
    i=0
    object_dict = {}
    while True:
        
        start_time = time.time()
        '''
        *********************** ARUCO  ***********************
        '''
        raw_frame = percept_in_conn.recv()
        frame_detect = raw_frame.copy()
        pd.detect_tags_3D(frame_detect)
        plane_dims = {'h':pd.plane_h, 'w':pd.plane_w}
        image_point_dict = pd.box_verts_update

        homography = pd.compute_homog(w_updated_pts=True, w_up_plane=True)
        warp = pd.compute_perspective_trans(raw_frame, w_updated_pts=True, w_up_plane=True)
        
        '''
        *********************** AI PART ***********************
        '''

        if warp is not None:
            boxes = detection(warp, model, disk_centroid_templates)
            #boxes = detection(warp, model)
            boxes, object_dict = object_tracking(object_dict, boxes)
            torch.cuda.empty_cache()
            frame_detect, data = plot_boxes(warp, 
                                            frame_detect, 
                                            boxes, 
                                            image_point_dict, 
                                            -1, 
                                            homography, 
                                            config['plane']['corners'],
                                            class_names=class_names, 
                                            plane_dims=plane_dims)
            percept_out_conn.send(data)
            warp = cv2.resize(warp, (warp.shape[1]*4,warp.shape[0]*4))
            cv2.imshow('warp', warp)
        img_scaling = 2.3
        cv2.imshow('frame', cv2.resize(frame_detect,(int(frame_detect.shape[1]*img_scaling),int(frame_detect.shape[0]*img_scaling))))

        key = cv2.waitKey(1)
        
        if key == 27:
            break

        if key == ord('s'):
            i+=1
            cv2.imwrite('images/raw_frame'+str(i) +'.png', raw_frame)
            cv2.imwrite('images/warp_frame'+str(i) +'.png', warp)
            
        print("FPS:" + str(1.0 / (time.time() - start_time)))

def post_detections(send_detect_in_conn, url_detections, is_post):
    while True:
        data = send_detect_in_conn.recv()
        try:
            if is_post:
                server_return = requests.post(url_detections, json=data)
                print('[INFO]: Detections posted.')
        except:
            break

if __name__ == '__main__':

    mp.set_start_method("spawn")
    IS_ONLINE = False
    TAG_TYPE = 'april'
    CAM_TYPE = 'rpi'
    MODEL_TYPE = 'mobnet'
    MODEL_PATH = './model_configs/'
    CAM_CONFIG_PATH = './vision_configs/'
    url_detections = 'http://10.41.0.2:5000/detections'
    MODEL = 'MOBILENET_V2_FINER_GRID_2_weights_saved.pt'if MODEL_TYPE == 'mobnet' else 'RESNET_18_FINER_GRID_2_weights_saved.pt'
    cam_source = 'http://10.41.0.2:8080/?action=stream' if IS_ONLINE else 'assets/delta_robot.mp4'
    
    path_dict = {
    'cam_matrix':{'rpi':CAM_CONFIG_PATH+'camera_matrix_rpi.txt',
                    'pc':CAM_CONFIG_PATH+'camera_matrix_pc_cam.txt' },

    'distortion':{'rpi':CAM_CONFIG_PATH+'distortion_rpi.txt',
                    'pc':CAM_CONFIG_PATH+'distortion_pc_cam.txt' },

    'plane_pts':{'april':CAM_CONFIG_PATH+'plane_points_new_tray.json',
                    'aruco':CAM_CONFIG_PATH+'plane_points_old_tray.json'},

    'model' : {'model_config':MODEL_PATH+MODEL,
                    'weights':None},
        }

    plane_config = {
    'tag_dicts' : {'aruco':cv2.aruco.DICT_4X4_50,
                'april':cv2.aruco.DICT_APRILTAG_36h11},
    
    'plane_corners' : {'aruco': {'tl' :'0','tr' :'1','br' :'2','bl' :'3'}, 
                        'april':{'tl' :'30','tr' :'101','br' :'5','bl' :'6'}},
        }

    config = {
    'neural_net': path_dict['model'],

    'vision': (path_dict['cam_matrix'][CAM_TYPE],
                path_dict['distortion'][CAM_TYPE],
                path_dict['plane_pts'][TAG_TYPE]),

    'plane': {'tag_size' : 2.86,
            'tag_scaling' : 0.36,
            'z_tansl' : 2.55,
            'tag_dict': plane_config['tag_dicts'][TAG_TYPE],
            'corners': plane_config['plane_corners'][TAG_TYPE]}
        }

    percept_in_conn, cam_out_conn = Pipe()
    send_detect_in_conn, percept_out_conn = Pipe()

    stream_reader_process = Process(target=cam_reader, 
                                    args=(cam_out_conn, cam_source))
    rob_percept_process = Process(target=robot_perception, 
                                    args=(percept_in_conn, percept_out_conn, config))
    post_detect_process = Process(target=post_detections, 
                                    args=(send_detect_in_conn,url_detections,IS_ONLINE))
    # start the receiver
    stream_reader_process.start()
    rob_percept_process.start()
    post_detect_process.start()
    # wait for all processes to finish
    rob_percept_process.join()
    stream_reader_process.kill()
    post_detect_process.kill()
    