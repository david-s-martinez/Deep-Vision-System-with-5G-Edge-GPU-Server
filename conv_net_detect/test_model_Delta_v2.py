import copy
import torch
import cv2
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
ANCHOR_BOXES = [[3.8] * 2]
GRID_SIZE_WIDTH = 18
GRID_SIZE_HEIGHT = 12
DEVICE = 0 if torch.cuda.is_available() else "cpu"
NUMBER_CLASSES = 3
CONFIDENCE_THRESHOLD = 0.8
IOU_THRESHOLD = 0.1
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 270


def plot_bbox(frame, bbox, is_prediction=True):
    x_min, y_min, x_max, y_max, object_class, confidence_score, centre_x, centre_y = int(IMAGE_WIDTH * bbox[0]), int(IMAGE_HEIGHT * bbox[1]), \
                                                                                     int(IMAGE_WIDTH * bbox[2]), int(IMAGE_HEIGHT * bbox[3]), \
                                                                                     bbox[5], bbox[4], \
                                                                                     int(IMAGE_WIDTH * bbox[6]), int(IMAGE_HEIGHT * bbox[7])
    print(x_min, y_min, x_max, y_max)
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

            predicted_bboxes_single_image = [bbox for bbox in predicted_bboxes_single_image if compute_IOU(torch.tensor(picked_bbox[3:]), torch.tensor(bbox[3:])) < IOU_THRESHOLD or bbox[1] !=
                                             picked_bbox[1]]

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
    plt.imshow(object_cropped)
    plt.show()
    gray = cv2.cvtColor(object_cropped, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_area = list()
    for contour in contours:
        contours_area.append(cv2.contourArea(contour))

    object_contour = contours[contours_area.index(max(contours_area))]
    center_x, center_y = compute_contour_centre(object_contour)

    return (center_x + x_min)/IMAGE_WIDTH, (center_y + y_min)/IMAGE_HEIGHT


def increase_contrast(image):
    transformations = A.Compose([
            A.RandomContrast(p=1, limit=(0.25, 0.2500001))
    ])

    return transformations(image=image)["image"]


def bbox_circles(frame, bbox):

    x_min, y_min, x_max, y_max, predicted_class = int(bbox[0] * IMAGE_WIDTH), int(bbox[1] * IMAGE_HEIGHT), int(bbox[2] * IMAGE_WIDTH), int(bbox[3] * IMAGE_HEIGHT), bbox[4]
    if predicted_class == 1:
        x_min = x_min + 5
        x_max -= 5
        y_min += 5
        y_max -= 5
    object_cropped = copy.deepcopy(frame[y_min:y_max, x_min:x_max, ...])

    gray = cv2.cvtColor(object_cropped, cv2.COLOR_BGR2GRAY)
    gray = np.asarray(gray, dtype=np.uint8)
    gray = increase_contrast(gray)

    if predicted_class == 0:
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_area = list()
        for contour in contours:
            contours_area.append(cv2.contourArea(contour))

        object_contour = contours[contours_area.index(max(contours_area))]
        center_x, center_y = compute_contour_centre(object_contour)
    else:
        if predicted_class == 1:
            min_radius, max_radius = int((x_max - x_min)/2 * 0.99), int((x_max - x_min)/2)
        else:
            min_radius, max_radius = int((x_max - x_min) / 2) - 5, int((x_max - x_min) / 2)
        circles_img = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 60,
                                      param1=300, param2=0.99,
                                       minRadius=min_radius, maxRadius=max_radius)
        circles_img = np.uint16(np.around(circles_img))
        center_x, center_y = circles_img[0][0][0], circles_img[0][0][1]

        for i in circles_img[0, :]:
            cv2.circle(object_cropped, (i[0], i[1]), i[2], (0, 255, 0), 1)
            cv2.circle(object_cropped, (i[0], i[1]), 2, (0, 0, 255), 1)

    return (center_x + x_min)/IMAGE_WIDTH, (center_y + y_min)/IMAGE_HEIGHT


def get_bboxes(predicted_bboxes, image_index):

    bboxes_predicted_relative_image = convert_cell_to_image(predicted_bboxes[..., 4:])
    predicted_class = predicted_bboxes[..., :NUMBER_CLASSES].argmax(-1).unsqueeze(-1)
    predicted_bboxes_modified = torch.cat([predicted_class, predicted_bboxes[..., NUMBER_CLASSES:NUMBER_CLASSES + 1], bboxes_predicted_relative_image], dim=-1)
    predicted_bboxes_list = get_bboxes_list(predicted_bboxes_modified, is_predictions=True, index=image_index)
    predicted_bbox_after_non_max_suppression = non_maxima_suppression(predicted_bboxes_list)

    return predicted_bbox_after_non_max_suppression


def make_prediction(frame_normalized, frame, model):
    predictions = model(frame_normalized)

    predictions[..., 4:6] = modified_sigmoid(predictions[..., 4:6], coefficient=0.75)
    predictions[..., 6:] = torch.tensor(ANCHOR_BOXES).reshape(1, 1, 1, 2).to(DEVICE) * modified_sigmoid(predictions[..., 6:], coefficient=0.75)
    bboxes = get_bboxes(predicted_bboxes=predictions, image_index=0)[0]
    bboxes_to_return = list()
    for bbox in bboxes:
        x_min = bbox[3] - bbox[5]/2
        y_min = bbox[4] - bbox[6]/2
        x_max = bbox[3] + bbox[5]/2
        y_max = bbox[4] + bbox[6]/2
        confidence_score = bbox[2] if bbox[2] < 1.0 else 1.0
        predicted_class = bbox[1]

        object_centroid = bbox_circles(frame, [x_min, y_min, x_max, y_max, predicted_class])
        bboxes_to_return.append([x_min, y_min, x_max, y_max, confidence_score, predicted_class, object_centroid[0], object_centroid[1]])
    return bboxes_to_return


def detection(frame, model, visualize_detection=False):
    frame = copy.deepcopy(reshape(frame, w=IMAGE_WIDTH, h=IMAGE_HEIGHT))
    frame_normalized = image_normalization(image=frame).unsqueeze(0).float().to(DEVICE)
    bboxes = make_prediction(frame_normalized, frame, model)
    if visualize_detection:
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        for bbox in bboxes:
            frame = plot_bbox(frame, bbox)
        plt.imshow(frame/255)
        plt.show()
    return bboxes


if __name__ == "__main__":
    frame = cv2.imread("/Users/artemmoroz/Desktop/CIIRC_projects/Detection_artificial_dataset_v3/warp_images_test_segmentation/warp_frame1.png")
    model = torch.load("/Users/artemmoroz/Desktop/CIIRC_projects/Detection_artificial_dataset_v3/RESNET_18_saved.pt", map_location=torch.device(DEVICE))
    model.eval()
    bboxes = detection(frame=frame, model=model, visualize_detection=True)
