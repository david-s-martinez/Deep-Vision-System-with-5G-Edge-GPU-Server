import copy
import torch
import cv2
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

import torch
import torch.nn
import torchvision
#from torchvision.models import ResNet18_Weights, ResNet50_Weights, MobileNet_V2_Weights
# from CONFIG import DROPOUT
DROPOUT=0.0

class DetectionModel(torch.nn.Module):
	def __init__(self, number_classes, grid_size_width, grid_size_height, chosen_model):
		super(DetectionModel, self).__init__()
		self.number_classes = number_classes
		self.grid_size_width = grid_size_width
		self.grid_size_height = grid_size_height
		self.chosen_model = chosen_model
		if chosen_model == 0:
			self.convolutional_layers = self._create_convolutional_layers_ResNet18()
			self.fully_connected_layers = self._create_fully_connected_layers(grid_size_width=self.grid_size_width, grid_size_height=self.grid_size_height, number_classes=self.number_classes,
			                                                                  num_input_params=512)
		elif chosen_model == 1:
			self.convolutional_layers = self._create_convolutional_layers_ResNet34()
			self.fully_connected_layers = self._create_fully_connected_layers(grid_size_width=self.grid_size_width, grid_size_height=self.grid_size_height, number_classes=self.number_classes,
			                                                                  num_input_params=2048)
		elif chosen_model == 2:
			self.convolutional_layers = self._create_convolutional_layers_MobileNetV2()
			self.fully_connected_layers = self._create_fully_connected_layers(grid_size_width=self.grid_size_width, grid_size_height=self.grid_size_height, number_classes=self.number_classes,
			                                                                  num_input_params=1280)

	def forward(self, x):
		return self.fully_connected_layers(self.convolutional_layers(x)).reshape((-1, self.grid_size_height, self.grid_size_width, self.number_classes + 5))

	@staticmethod
	def _create_convolutional_layers_ResNet18():
		resnet18 = torchvision.models.resnet18()#(pretrained=True)#(weights=ResNet18_Weights.IMAGENET1K_V1)
		resnet18_layers = torch.nn.Sequential(*list(resnet18.children())[:-1])
		return resnet18_layers

	@staticmethod
	def _create_convolutional_layers_ResNet34():
		resnet50 = torchvision.models.resnet50()#(pretrained=True)#(weights=ResNet50_Weights.IMAGENET1K_V2)
		resnet50_layers = torch.nn.Sequential(*list(resnet50.children())[:-1])
		return resnet50_layers

	@staticmethod
	def _create_convolutional_layers_MobileNetV2():
		mobilenetV2 = torchvision.models.mobilenet_v2()#(pretrained=True)#(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
		mobilenetV2_layers = torch.nn.Sequential(*list(mobilenetV2.children())[:-1], torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

		return mobilenetV2_layers

	@staticmethod
	def _create_fully_connected_layers(grid_size_width, grid_size_height, number_classes, num_input_params):
		return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(num_input_params, 1024), torch.nn.Dropout(DROPOUT),
		                           torch.nn.LeakyReLU(0.1), torch.nn.Linear(1024, grid_size_width * grid_size_height * (number_classes + 5)))


ANCHOR_BOXES = [[5.5] * 2]
GRID_SIZE_WIDTH = 27
GRID_SIZE_HEIGHT = 18
DEVICE = 0 if torch.cuda.is_available() else "cpu"
NUMBER_CLASSES = 3
CONFIDENCE_THRESHOLD = 0.8
IOU_THRESHOLD = 0.1
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 270


def compute_iou(bbox_1, bbox_2):
	x_min = max(bbox_1[0], bbox_2[0])
	y_min = max(bbox_1[1], bbox_2[1])
	x_max = min(bbox_1[2], bbox_2[2])
	y_max = min(bbox_1[3], bbox_2[3])

	height = (y_max - y_min) if (y_max - y_min) > 0 else 0
	width = (x_max - x_min) if (x_max - x_min) > 0 else 0

	intersection = height * width

	union = np.abs((bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1])) + \
	        np.abs((bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])) - \
	        intersection

	return intersection/(union + 0.00001)


def object_tracking(objects_dict, boxes):
    if len(objects_dict.keys()) == 0:
        for index in range(len(objects_dict.keys()), len(boxes)):
            objects_dict[str(index)] = boxes[index]
    elif len(objects_dict.keys()) < len(boxes) and len(objects_dict.keys()) != 0:
        new_instances_to_add = list()
        for key, value in objects_dict.items():
            for bbox in boxes:
                iou = compute_iou(bbox_1=[bbox[0], bbox[1], bbox[2], bbox[3]], bbox_2=[value[0], value[1], value[2], value[3]])
                if iou < 0.25:
                    new_instances_to_add.append(bbox)
        for index in range(len(objects_dict.keys()), len(objects_dict.keys()) + len(new_instances_to_add)):
            objects_dict[str(index)] = new_instances_to_add[index - len(objects_dict.keys())]
    elif len(boxes) < len(objects_dict.keys()):
        key_list = list(objects_dict.keys())
        values_list = list(objects_dict.values())

        for index_1 in range(len(values_list)):
            for index_2 in range(index_1, len(values_list)):
                iou = compute_iou(bbox_1=[values_list[index_1][0], values_list[index_1][1],
                                          values_list[index_1][2], values_list[index_1][3]],
                                  bbox_2=[values_list[index_2][0], values_list[index_2][1],
                                          values_list[index_2][2], values_list[index_2][3]])
                if iou > 0.9 and key_list[index_2] in objects_dict.keys():
                    objects_dict.pop(key_list[index_2])

        keys_to_preserve = list()
        for key, value in objects_dict.items():
            for bbox in boxes:
                iou = compute_iou(bbox_1=[bbox[0], bbox[1], bbox[2], bbox[3]], bbox_2=[value[0], value[1], value[2], value[3]])
                if iou >= 0.1:
                    keys_to_preserve.append(key)
                    break
        keys_to_delete = list()
        for key in objects_dict.keys():
            if key not in keys_to_preserve:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            objects_dict.pop(key)
    elif len(boxes) == len(objects_dict.keys()) and len(boxes) > 0:
        list_keys = list()
        list_boxes = list()
        for key in objects_dict.keys():
            for box in boxes:
                iou = compute_iou(bbox_1=[box[0], box[1], box[2], box[3]],
                                  bbox_2=[objects_dict[key][0], objects_dict[key][1],
                                          objects_dict[key][2], objects_dict[key][3]])
                if iou > 0.5:
                    list_boxes.append(box)
                    list_keys.append(key)
                    break
        if len(list_boxes) < len(boxes):
            key_not_in_list = None
            for key in objects_dict.keys():
                if key not in list_keys:
                    key_not_in_list = key
            box_not_in_list = None
            for box in boxes:
                if box not in list_boxes:
                    box_not_in_list = box

            objects_dict[key_not_in_list] = box_not_in_list
    beta = 0.95
    for key, value in objects_dict.items():
        for bbox in boxes:
            iou = compute_iou(bbox_1=[bbox[0], bbox[1], bbox[2], bbox[3]], bbox_2=[value[0], value[1], value[2], value[3]])
            if iou >= 0.1:
                center_x = objects_dict[key][-2] * beta + bbox[-2] * (1 - beta)
                center_y = objects_dict[key][-1] * beta + bbox[-1] * (1 - beta)
                x_min = objects_dict[key][0] * beta + bbox[0] * (1 - beta)
                y_min = objects_dict[key][1] * beta + bbox[1] * (1 - beta)
                x_max = objects_dict[key][2] * beta + bbox[2] * (1 - beta)
                y_max = objects_dict[key][3] * beta + bbox[3] * (1 - beta)
                predicted_class = objects_dict[key][5] * beta + bbox[5] * (1 - beta)
                objects_dict[key] = [x_min, y_min, x_max, y_max, bbox[4], predicted_class, center_x, center_y]
                break
    boxes = [value for value in objects_dict.values()]

    return boxes, objects_dict


def plot_bbox(frame, bbox, is_prediction=True):
    x_min, y_min, x_max, y_max, object_class, confidence_score, centre_x, centre_y = int(IMAGE_WIDTH * bbox[0]), int(IMAGE_HEIGHT * bbox[1]), \
                                                                                     int(IMAGE_WIDTH * bbox[2]), int(IMAGE_HEIGHT * bbox[3]), \
                                                                                     bbox[5], bbox[4], \
                                                                                     int(IMAGE_WIDTH * bbox[6]), int(IMAGE_HEIGHT * bbox[7])
    if confidence_score > 1.0:
        confidence_score = 1.0
    if is_prediction:
        if int(object_class) == 0:
            object_class = "tyre"
        elif int(object_class) == 2:
            object_class = "wheel"
        elif int(object_class) == 1:
            object_class = "disk"

        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
        frame[centre_y - 1:centre_y + 1, centre_x - 1:centre_x + 1, :] = [255, 255, 0]
        frame = cv2.putText(frame, object_class + " " + str(round(confidence_score, ndigits=2)), (x_min, y_min + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    return frame


def reshape(image, w, h):
    reshaped_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    return reshaped_image


def compute_IOU(bboxes1, bboxes2):

    x1min = bboxes1[..., 0:1] - bboxes1[..., 2:3] / 2
    y1min = bboxes1[..., 1:2] - bboxes1[..., 3:4] / 2
    x1max = bboxes1[..., 0:1] + bboxes1[..., 2:3] / 2
    y1max = bboxes1[..., 1:2] + bboxes1[..., 3:4] / 2

    x2min = bboxes2[..., 0:1] - bboxes2[..., 2:3] / 2
    y2min = bboxes2[..., 1:2] - bboxes2[..., 3:4] / 2
    x2max = bboxes2[..., 0:1] + bboxes2[..., 2:3] / 2
    y2max = bboxes2[..., 1:2] + bboxes2[..., 3:4] / 2

    Xmin = torch.max(x1min, x2min)
    Ymin = torch.max(y1min, y2min)
    Xmax = torch.min(x1max, x2max)
    Ymax = torch.min(y1max, y2max)

    intersection = (Xmax - Xmin).clamp(0) * (Ymax - Ymin).clamp(0)

    union = torch.abs((x1max - x1min) * (y1max - y1min)) + torch.abs((x2max - x2min) * (y2max - y2min)) - intersection

    return intersection/(union + 0.0000001)


def modified_sigmoid(x, coefficient=0.5):
    return 1 / (1 + torch.exp(-coefficient * x))


def get_bboxes_list(bboxes, index=0, is_predictions=False):
    bboxes_list = list()

    bboxes_reshaped = bboxes.reshape(bboxes.shape[0], bboxes.shape[1] * bboxes.shape[2], -1)

    index_counter = 0
    for image_index in range(bboxes.shape[0]):
        image_bboxes = list()
        for cell_index in range(bboxes_reshaped.shape[1]):
            if not is_predictions and not torch.all(bboxes_reshaped[image_index, cell_index, :].eq(torch.zeros(6).to(DEVICE))):
                image_bboxes.append([index + index_counter] + bboxes_reshaped[image_index, cell_index, :].tolist())
            elif is_predictions:
                image_bboxes.append([index + index_counter] + bboxes_reshaped[image_index, cell_index, :].tolist())
        index_counter += 1
        bboxes_list.append(image_bboxes)

    return bboxes_list


def non_maxima_suppression(predicted_bboxes):
    predictions_after_non_max_suppression = list()

    for image_index in range(len(predicted_bboxes)):
        predicted_bboxes_single_image = predicted_bboxes[image_index]

        single_image_predictions_non_max_suppression = list()

        predicted_bboxes_single_image = [bbox for bbox in predicted_bboxes_single_image if bbox[2] > CONFIDENCE_THRESHOLD]
        predicted_bboxes_single_image = sorted(predicted_bboxes_single_image, reverse=True, key=lambda x: x[2])

        while predicted_bboxes_single_image:
            picked_bbox = predicted_bboxes_single_image.pop(0)

            predicted_bboxes_single_image = [bbox for bbox in predicted_bboxes_single_image if compute_IOU(torch.tensor(picked_bbox[3:]), torch.tensor(bbox[3:])) < IOU_THRESHOLD] #or bbox[1] !=
                                                #picked_bbox[1]]

            single_image_predictions_non_max_suppression.append(picked_bbox)
        predictions_after_non_max_suppression.append(single_image_predictions_non_max_suppression)

    return predictions_after_non_max_suppression


def convert_cell_to_image(bboxes):

    cell_indicis_x = torch.arange(GRID_SIZE_WIDTH).repeat(bboxes.shape[0], GRID_SIZE_HEIGHT, 1).unsqueeze(-1)
    cell_indicis_y = torch.arange(GRID_SIZE_HEIGHT).repeat(bboxes.shape[0], GRID_SIZE_WIDTH, 1).unsqueeze(-1)
    cell_indicis_y = cell_indicis_y.permute(0, 2, 1, 3)
    bboxes[..., :1] = (bboxes[..., :1] + cell_indicis_x.to(DEVICE))/GRID_SIZE_WIDTH
    bboxes[..., 1:2] = (bboxes[..., 1:2] + cell_indicis_y.to(DEVICE))/GRID_SIZE_HEIGHT
    bboxes[..., 2:3] = bboxes[..., 2:3] / GRID_SIZE_WIDTH
    bboxes[..., 3:4] = bboxes[..., 3:4] / GRID_SIZE_HEIGHT
    return bboxes


def image_normalization(image):
    testing_preprocessing = A.Compose(
        [
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2()
        ]
    )
    return testing_preprocessing(image=image)["image"]


def compute_contour_centre(contour):
    centre_x, centre_y = 0, 0
    for i in range(len(contour)):
        centre_x += contour[i][0][0]
        centre_y += contour[i][0][1]
    return int(centre_x/len(contour)), int(centre_y / len(contour))


def bbox_contours(frame, bbox):
    x_min, y_min, x_max, y_max = int(bbox[0] * IMAGE_WIDTH), int(bbox[1] * IMAGE_HEIGHT), int(bbox[2] * IMAGE_WIDTH), int(bbox[3] * IMAGE_HEIGHT)

    object_cropped = frame[y_min:y_max, x_min:x_max, ...]

    gray = cv2.cvtColor(object_cropped, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_area = list()
    for contour in contours:
        contours_area.append(cv2.contourArea(contour))

    object_contour = contours[contours_area.index(max(contours_area))]
    center_x, center_y = compute_contour_centre(object_contour)
   # y_shift = (80 - y_min) * 0.15 if y_min <= 80 else (y_min - 80) * 0.15
    return (center_x + x_min)/IMAGE_WIDTH, (center_y + y_min)/IMAGE_HEIGHT


def increase_contrast(image):
    transformations = A.Compose([
            A.RandomContrast(p=1, limit=(0.25, 0.2500001))
    ])

    return transformations(image=image)["image"]


def find_centroids(frame, bbox, templates):

    x_min, y_min, x_max, y_max, predicted_class = int(bbox[0] * IMAGE_WIDTH), int(bbox[1] * IMAGE_HEIGHT), int(bbox[2] * IMAGE_WIDTH), int(bbox[3] * IMAGE_HEIGHT), bbox[4]
    object_cropped = copy.deepcopy(frame[y_min:y_max, x_min:x_max, ...])
    #print(np.mean(object_cropped) )
    if np.mean(object_cropped) > 100:
        return None
    object_cropped = increase_contrast(object_cropped)
    gray = cv2.cvtColor(object_cropped, cv2.COLOR_BGR2GRAY)
    gray = np.asarray(gray, dtype=np.uint8)
    gray = increase_contrast(gray)

    if predicted_class in [0]:
        gray = cv2.Canny(gray, 30, 50, apertureSize=3)
        circles_img = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 60,
                                       param1=300, param2=1, minRadius=18, maxRadius=22)
        if (circles_img is not None):
            center_x = int(circles_img[0][0][0])
            center_y = int(circles_img[0][0][1])
        else:
            center_x = int((x_max + x_min) / 2)
            center_y = int((y_max + y_min) / 2)
        '''ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_area = list()
        for contour in contours:
            contours_area.append(cv2.contourArea(contour))

        object_contour = contours[contours_area.index(max(contours_area))]

        center_x, center_y = compute_contour_centre(object_contour)'''
    elif predicted_class in [2]:
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

        #contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if predicted_class == 1:
            min_rad, max_rad = 20, 25
        else:
            min_rad, max_rad = 25, 30
        circles_img = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 60,
                                       param1=300, param2=1, minRadius=min_rad, maxRadius=max_rad)
        #print(circles_img)
        center_x_1 = 0
        center_y_1 = 0
        if (circles_img is not None):
            for circ in circles_img:
                center_x_1 += int(circ[0][0])
                center_y_1 += int(circ[0][1])
            center_y_1 /= len(circles_img)
            center_x_1 /= len(circles_img)
        else:
            center_x_1 = int((x_max + x_min) / 2)
            center_y_1 = int((y_max + y_min) / 2)
    #elif predicted_class in [1, 2]:
        methods_names = ["cv2.TM_SQDIFF_NORMED", "cv2.TM_CCORR_NORMED", 'cv2.TM_CCOEFF_NORMED']
        center_x_2 = 0
        center_y_2 = 0
        for method_name in methods_names:
            method = eval(method_name)  # eval("TM_CCORR_NORMED")
            match_values = list()
            for template in templates:
                result = cv2.matchTemplate(object_cropped, template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if method_name == "cv2.TM_SQDIFF_NORMED":
                    match_values.append((min_val, min_loc))
                else:
                    match_values.append((max_val, max_loc))
            if method_name == "cv2.TM_SQDIFF_NORMED":
                match_values.sort(reverse=False, key=lambda x: x[0])
            else:
                match_values.sort(reverse=True, key=lambda x: x[0])
            top_left = match_values[0][1]
            bottom_right = (top_left[0] + templates[0].shape[1], top_left[1] + templates[0].shape[0])
            center_x_2 += int((top_left[0] + bottom_right[0]) / 2)
            center_y_2 += int((top_left[1] + bottom_right[1]) / 2)
        center_y_2 /= 3
        center_x_2 /= 3
        center_x = center_x_1 + center_x_2
        center_y = center_y_1 + center_y_2
        center_x /= 2
        center_y /= 2
    elif predicted_class in [1]:
        methods_names = ["cv2.TM_SQDIFF_NORMED", "cv2.TM_CCORR_NORMED", 'cv2.TM_CCOEFF_NORMED']
        center_x_2 = 0
        center_y_2 = 0
        for method_name in methods_names:
            method = eval(method_name)  # eval("TM_CCORR_NORMED")
            match_values = list()
            for template in templates:
                result = cv2.matchTemplate(object_cropped, template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if method_name == "cv2.TM_SQDIFF_NORMED":
                    match_values.append((min_val, min_loc))
                else:
                    match_values.append((max_val, max_loc))
            if method_name == "cv2.TM_SQDIFF_NORMED":
                match_values.sort(reverse=False, key=lambda x: x[0])
            else:
                match_values.sort(reverse=True, key=lambda x: x[0])
            top_left = match_values[0][1]
            bottom_right = (top_left[0] + templates[0].shape[1], top_left[1] + templates[0].shape[0])
            center_x_2 += int((top_left[0] + bottom_right[0]) / 2)
            center_y_2 += int((top_left[1] + bottom_right[1]) / 2)
        center_x = center_x_2 / 3
        center_y = center_y_2 / 3
    return (center_x + x_min)/IMAGE_WIDTH, (center_y + y_min)/IMAGE_HEIGHT


def euclidean_distance(point_1, point_2):
    x_1 = point_1[0] * IMAGE_WIDTH
    x_2 = point_2[0] * IMAGE_WIDTH
    y_1 = point_1[1] * IMAGE_HEIGHT
    y_2 = point_2[1] * IMAGE_HEIGHT
    return np.sqrt((x_2 - x_1)**2 + (y_1 - y_2)**2)


def get_bboxes(predicted_bboxes, image_index):

    bboxes_predicted_relative_image = convert_cell_to_image(predicted_bboxes[..., 4:])
    predicted_class = predicted_bboxes[..., :NUMBER_CLASSES].argmax(-1).unsqueeze(-1)
    confidence_score = predicted_bboxes[..., NUMBER_CLASSES:NUMBER_CLASSES + 1]
    class_probs, _ = torch.max(torch.max(torch.softmax(predicted_bboxes[..., :NUMBER_CLASSES], -1), confidence_score), -1)
    #print(class_probs.shape)
    #print(predicted_bboxes[..., :NUMBER_CLASSES])
    #print(class_probs.shape)
    #print(confidence_score.shape)
    predicted_bboxes_modified = torch.cat([predicted_class, confidence_score, bboxes_predicted_relative_image], dim=-1)
    predicted_bboxes_list = get_bboxes_list(predicted_bboxes_modified, is_predictions=True, index=image_index)
    predicted_bbox_after_non_max_suppression = non_maxima_suppression(predicted_bboxes_list)

    return predicted_bbox_after_non_max_suppression


def make_prediction(frame_normalized, frame, model, templates):
    predictions = model(frame_normalized)

    predictions[..., 4:6] = modified_sigmoid(predictions[..., 4:6], coefficient=1)
    predictions[..., 6:] = torch.tensor(ANCHOR_BOXES).reshape(1, 1, 1, 2).to(DEVICE) * modified_sigmoid(predictions[..., 6:], coefficient=1)
    bboxes = get_bboxes(predicted_bboxes=predictions, image_index=0)[0]
    bboxes_to_return = list()
    for bbox in bboxes:
        if bbox[3] >= 0.88 or bbox[3] <= 0.12 or bbox[4] <= 0.21 or bbox[4] >= 0.93:
            continue
        predicted_class = bbox[1]
        if predicted_class == 1:
            bbox[5] *= 0.9
            bbox[6] *= 0.9
        x_min = bbox[3] - bbox[5]/2
        y_min = bbox[4] - bbox[6]/2
        x_max = bbox[3] + bbox[5]/2
        y_max = bbox[4] + bbox[6]/2
        confidence_score = bbox[2] if bbox[2] < 1.0 else 1.0

        object_centroid = find_centroids(frame, [x_min, y_min, x_max, y_max, predicted_class], templates)
        if object_centroid is None:
            continue
        if euclidean_distance(object_centroid, (bbox[3], bbox[4])) > 25 and predicted_class != 0:
            bboxes_to_return.append([x_min, y_min, x_max, y_max, confidence_score, predicted_class, (bbox[3]), (bbox[4])])
        else:
            bboxes_to_return.append([x_min, y_min, x_max, y_max, confidence_score, predicted_class, object_centroid[0], object_centroid[1]])
    return bboxes_to_return


def detection(frame, model, templates, visualize_detection=False):
    frame = copy.deepcopy(reshape(frame, w=IMAGE_WIDTH, h=IMAGE_HEIGHT))
    frame_normalized = image_normalization(image=frame).unsqueeze(0).float().to(DEVICE)
    bboxes = make_prediction(frame_normalized, frame, model, templates)
    if visualize_detection:
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        for bbox in bboxes:
            frame = plot_bbox(frame, bbox)
        plt.imshow(frame/255)
        plt.show()
    return bboxes


if __name__ == "__main__":
    disk_centroid_templates = [cv2.imread("disk_centroid_template_1.png"), cv2.imread("disk_centroid_template_2.png"), cv2.imread("disk_centroid_template_3.png")]
    frame = cv2.imread("/Users/artemmoroz/Desktop/CIIRC_projects/Detection_artificial_dataset_v3/warp_images_test_segmentation/warp_frame520.png")
    model = Detection_models.DetectionModel(number_classes=NUMBER_CLASSES, grid_size_width=GRID_SIZE_WIDTH, grid_size_height=GRID_SIZE_HEIGHT, chosen_model=0)
    model.load_state_dict(torch.load("/Users/artemmoroz/Desktop/CIIRC_projects/Detection_artificial_dataset_v3/RESNET_18_FINER_GRID_2_weights_saved.pt", map_location="cpu"))
    #model = torch.load("/Users/artemmoroz/Desktop/CIIRC_projects/Detection_artificial_dataset_v3/RESNET_18_saved.pt", map_location=torch.device(DEVICE))
    model.eval()
    bboxes = detection(frame=frame, model=model, templates=disk_centroid_templates, visualize_detection=True)
