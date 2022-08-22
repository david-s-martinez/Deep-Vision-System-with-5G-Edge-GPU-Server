import time

import cv2
import numpy as np
import albumentations as A
import torch
import matplotlib.pyplot as plt
from collections import Counter
from CONFIG import GRID_SIZE_WIDTH, NUMBER_CLASSES, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT, ANCHOR_BOXES, LEARNING_RATE, MODEL_NAME, GRID_SIZE_HEIGHT, ROOT_PATH
from Detection_train_model import DEVICE
from PIL import Image


def remove_background(image):
    zeros = np.ones((image.shape[0], image.shape[1]))
    R_channel = image[..., 0] >= 150 * zeros
    G_channel = image[..., 1] >= 150 * zeros
    B_channel = image[..., 2] >= 150 * zeros
    black_mask = np.array([R_channel * G_channel * B_channel])
    base_image = np.ones(image.shape) * (np.array([255, 255, 255]))

    image_no_black = image * (1 - np.transpose(np.repeat(black_mask, [3], axis=0), axes=[1, 2, 0])) + base_image * np.transpose(np.repeat(black_mask, [3], axis=0), axes=[1, 2, 0])
    return image_no_black


def get_gripper_mask(gripper_image):
    zeros = 20 * np.ones((gripper_image.shape[0], gripper_image.shape[1]))
    R_channel = gripper_image[..., 0] >= zeros
    G_channel = gripper_image[..., 1] >= zeros
    B_channel = gripper_image[..., 2] >= zeros
    white_mask = np.array(([R_channel * G_channel * B_channel]))

    return white_mask


def remove_white(image):
    zeros = 180 * np.ones((image.shape[0], image.shape[1]))
    R_channel = image[..., 0] >= zeros
    G_channel = image[..., 1] >= zeros
    B_channel = image[..., 2] >= zeros
    white_mask = np.array(([R_channel * G_channel * B_channel]))
    base_image = np.ones(image.shape) * (np.array([200, 202, 202]))
    new_img = image * (1 - np.transpose(np.repeat(white_mask, [3], axis=0), axes=[1, 2, 0])) + base_image * np.transpose(np.repeat(white_mask, [3], axis=0), axes=[1, 2, 0])

    return new_img


def remove_black(image):
    zeros = 0 * np.ones((image.shape[0], image.shape[1]))
    R_channel = image[..., 0] <= zeros
    G_channel = image[..., 1] <= zeros
    B_channel = image[..., 2] <= zeros
    black_mask = np.array([R_channel * G_channel * B_channel])
    base_image = np.ones(image.shape) * (np.array([200, 202, 202]))

    image_after_black_mask = base_image * np.transpose(np.repeat(black_mask, [3], axis=0), axes=[1, 2, 0])

    image_no_black = image_after_black_mask + image

    return image_no_black


def init_model_weights(model):
    for lay in model.modules():
        if type(lay) in [torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear]:
            torch.nn.init.xavier_uniform(lay.weight)
    return model


def reshape(image, w, h):
    reshaped_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    return reshaped_image


def get_white_mask(image_white):
    zeros = 220 * np.ones((image_white.shape[0], image_white.shape[1]))
    R_channel = image_white[..., 0] >= zeros
    G_channel = image_white[..., 1] >= zeros
    B_channel = image_white[..., 2] >= zeros
    white_mask = np.array(([R_channel * G_channel * B_channel]))

    return white_mask


def save_perspective_transformation(white_mask, image, magnitude=0, is_disk=True):
    if magnitude == 1:
        multiplier = 1
        height = 224 if not is_disk else 190
    elif magnitude == 2:
        multiplier = 1.1
        height = 244 if not is_disk else 224
    elif magnitude == 3:
        multiplier = 1.2
        height = 264 if not is_disk else 244
    elif magnitude == 4:
        multiplier = 1.3
        height = 290 if not is_disk else 264
    width = 224 if not is_disk else 190

    translated_image = np.ones((int(image.shape[0] * multiplier), image.shape[1], 3)) * np.array([172, 172, 172])

    for x in range(image.shape[1]):
        counter_transformed = 0
        for y in range(image.shape[0]):
            if not white_mask[0, y, x]:
                translated_image[counter_transformed, x, :] = image[y, x, :]
                counter_transformed += 1
            else:
                translated_image[counter_transformed:counter_transformed + magnitude, x, :] = image[y, x, :]
                counter_transformed += magnitude

    translated_image = reshape(translated_image, width, height)
    if magnitude == 1:
        translated_image = translated_image[4:-8, 2:-2, :]
    elif magnitude == 2:
        translated_image = translated_image[4:-8, 2:, :]
    elif magnitude == 3:
        translated_image = translated_image[4:-19, 2:, :]
    elif magnitude == 4:
        translated_image = translated_image[4:-27, 2:, :]
    translated_image = remove_background(translated_image)
    plt.imshow(translated_image/255)
    plt.show()
    cv2.imwrite(ROOT_PATH + "images/disk_magnitude_{}.jpg".format(magnitude - 1), translated_image)

def rotate(image, angle):
    """
    Function implements image rotation
    :param image:
    :type ndarray
    :param angle:
    :type int
    :return: rotated_image
    :type ndarray
    """
    image = np.array(image, dtype=np.uint8)
    image = Image.fromarray(image)
    image = image.rotate(angle, fillcolor=(255, 255, 255), expand=True)
    image = np.asarray(image)

    return image

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


def plot_bboxes(images, predicted_bboxes, ground_truth_bboxes=None):

    for image_index in range(images.shape[0]):
        image = images[image_index, ...]
        if ground_truth_bboxes:
            for bbox in ground_truth_bboxes[image_index]:
                #print(bbox)
                image = plot_bbox(image, bbox, is_prediction=False)

        image_predicted_bboxes = predicted_bboxes

        for bbox in image_predicted_bboxes[image_index]:
            image = plot_bbox(image, bbox, is_prediction=True)

        plt.imshow(image)
        plt.show()
        #time.sleep(5)


def plot_bbox(image, bbox, is_prediction=False):
    image = np.ascontiguousarray(image, dtype=np.float32)
    index, object_class, confidence_score, x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6]

    x_relative_image, y_relative_image, w_relative_image, h_relative_image = x * IMAGE_SIZE_WIDTH, y * IMAGE_SIZE_HEIGHT, w * IMAGE_SIZE_WIDTH, h * IMAGE_SIZE_HEIGHT
    print(x_relative_image, y_relative_image, w_relative_image, h_relative_image)
    x_min = int(round(x_relative_image - w_relative_image / 2))
    y_min = int(round(y_relative_image - h_relative_image / 2))
    x_max = int(round(x_relative_image + w_relative_image / 2))
    y_max = int(round(y_relative_image + h_relative_image / 2))

    if confidence_score > 1.0:
        confidence_score = 1.0
    if is_prediction:
        if int(object_class) == 0:
            object_class = "tyre"
        elif int(object_class) == 2:
            object_class = "wheel"
        elif int(object_class) == 1:
            object_class = "disc"
        print((x_min, y_min), (x_max, y_max))
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
        image = cv2.putText(image, object_class + " " + str(round(confidence_score, ndigits=2)), (x_min, y_min + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    return image


def get_bboxes(predicted_bboxes, ground_truth_bboxes, image_index):

    bboxes_predicted_relative_image = convert_cell_to_image(predicted_bboxes[..., 4:])
    bboxes_ground_truth_relative_image = convert_cell_to_image(ground_truth_bboxes[..., 4:])

    predicted_class = predicted_bboxes[..., :NUMBER_CLASSES].argmax(-1).unsqueeze(-1)
    ground_truth_class = ground_truth_bboxes[..., :NUMBER_CLASSES].argmax(-1).unsqueeze(-1)

    predicted_bboxes_modified = torch.cat([predicted_class, predicted_bboxes[..., NUMBER_CLASSES:NUMBER_CLASSES + 1], bboxes_predicted_relative_image], dim=-1)
    ground_truth_bboxes_modified = torch.cat([ground_truth_class, ground_truth_bboxes[..., NUMBER_CLASSES:NUMBER_CLASSES + 1], bboxes_ground_truth_relative_image], dim=-1)

    ground_truth_confidence_mask = (ground_truth_bboxes[..., NUMBER_CLASSES:NUMBER_CLASSES + 1] == 1).repeat(1, 1, 1, 6)
    ground_truth_bboxes_modified = ground_truth_bboxes_modified * ground_truth_confidence_mask

    predicted_bboxes_list = get_bboxes_list(predicted_bboxes_modified, is_predictions=True, index=image_index)
    ground_truth_bboxes_list = get_bboxes_list(ground_truth_bboxes_modified, index=image_index)

    predicted_bbox_after_non_max_suppression = non_maxima_suppression(predicted_bboxes_list)

    return predicted_bbox_after_non_max_suppression, ground_truth_bboxes_list


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


def convert_cell_to_image(bboxes):

    cell_indicis_x = torch.arange(GRID_SIZE_WIDTH).repeat(bboxes.shape[0], GRID_SIZE_HEIGHT, 1).unsqueeze(-1)
    cell_indicis_y = torch.arange(GRID_SIZE_HEIGHT).repeat(bboxes.shape[0], GRID_SIZE_WIDTH, 1).unsqueeze(-1)
    cell_indicis_y = cell_indicis_y.permute(0, 2, 1, 3)
    #print(bboxes.shape)
    bboxes[..., :1] = (bboxes[..., :1] + cell_indicis_x.to(DEVICE))/GRID_SIZE_WIDTH
    bboxes[..., 1:2] = (bboxes[..., 1:2] + cell_indicis_y.to(DEVICE))/GRID_SIZE_HEIGHT
    bboxes[..., 2:3] = bboxes[..., 2:3] / GRID_SIZE_WIDTH
    bboxes[..., 3:4] = bboxes[..., 3:4] / GRID_SIZE_HEIGHT
    return bboxes


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


def change_learning_rate(optimizer, epoch):
    """if epoch <= 20:
        optimizer.param_groups[0]["lr"] = LEARNING_RATE * np.exp(epoch * np.log(50)/20)"""
    if epoch >= 40 and epoch % 20 == 0:
        optimizer.param_groups[0]["lr"] /= 10


def save_pretraining(model_state_dict):
    model_state_dict.pop("fully_connected_layers.4.weight")
    model_state_dict.pop("fully_connected_layers.4.bias")
    model_state_dict["fully_connected_layers.4.weight"] = torch.from_numpy(0.01 * np.random.randn(GRID_SIZE_WIDTH * GRID_SIZE_WIDTH * (NUMBER_CLASSES + 5), 4096))
    model_state_dict["fully_connected_layers.4.bias"] = torch.from_numpy(0.01 * np.random.randn(GRID_SIZE_WIDTH * GRID_SIZE_WIDTH * (NUMBER_CLASSES + 5)))
    torch.save(model_state_dict, "{}_CLASSIFICATION_PRETRAINED.pt".format(MODEL_NAME))


def mean_average_precision_iou_threshold(predicted_bboxes, ground_truth_bboxes, iou_threshold):
    predicted_bboxes = convert_bboxes_to_list(predicted_bboxes)
    ground_truth_bboxes = convert_bboxes_to_list(ground_truth_bboxes)

    average_precisions_per_class = list()

    for class_ in range(NUMBER_CLASSES):
        predicted_bboxes_class = list()
        ground_truth_bboxes_class = list()

        for bbox in predicted_bboxes:
            if int(bbox[1]) == class_:
                predicted_bboxes_class.append(bbox)

        for bbox in ground_truth_bboxes:
            if int(bbox[1]) == class_:
                ground_truth_bboxes_class.append(bbox)

        number_ground_truth_per_image = Counter([gt[0] for gt in ground_truth_bboxes_class])

        for key, value in number_ground_truth_per_image.items():
            number_ground_truth_per_image[key] = torch.zeros(value)

        predicted_bboxes_class.sort(reverse=True, key=lambda x: x[2])

        TP, FP = torch.zeros(len(predicted_bboxes_class)), torch.zeros(len(predicted_bboxes_class))
        TP_and_FN = len(ground_truth_bboxes_class)

        if TP_and_FN == 0:
            continue

        for pred_index, pred_bbox in enumerate(predicted_bboxes_class):
            ground_truth_bboxes_image = [gt_bbox for gt_bbox in ground_truth_bboxes_class if pred_bbox[0] == gt_bbox[0]]

            best_iou = 0
            best_iou_ground_truth_index = None

            for gt_index, gt_bbox in enumerate(ground_truth_bboxes_image):
                iou = compute_IOU(torch.tensor(pred_bbox[-4:]), torch.tensor(gt_bbox[-4:]))
                if iou > best_iou:
                    best_iou_ground_truth_index = gt_index
                    best_iou = iou

            if best_iou >= iou_threshold:
                if number_ground_truth_per_image[pred_bbox[0]][best_iou_ground_truth_index] == 0:
                    TP[pred_index] = 1
                    number_ground_truth_per_image[pred_bbox[0]][best_iou_ground_truth_index] = 1
                else:
                    FP[pred_index] = 1
            else:
                FP[pred_index] = 1

        TP_cumulative_sum = torch.cumsum(TP, dim=0)
        FP_cumulative_sum = torch.cumsum(FP, dim=0)
        precisions = TP_cumulative_sum / (TP_cumulative_sum + FP_cumulative_sum + 0.00001)
        recalls = TP_cumulative_sum / TP_and_FN
        precisions = torch.cat([torch.tensor([1]), precisions])
        recalls = torch.cat([torch.tensor([0]), recalls])
        average_precision = torch.trapz(precisions, recalls)
        average_precisions_per_class.append(average_precision)

    return sum(average_precisions_per_class)/len(average_precisions_per_class)


def convert_bboxes_to_list(bboxes):
    list_bboxes = list()
    for image in bboxes:
        for bbox in image:
            list_bboxes.append(bbox)

    return list_bboxes


def compute_mean_average_precision(predicted_bboxes, ground_truth_bboxes):
    iou_thresholds_mean_average_precision = list()
    print("start map computation")
    for iou in range(50, 95, 5):
        iou_thresholds_mean_average_precision.append(mean_average_precision_iou_threshold(predicted_bboxes=predicted_bboxes, ground_truth_bboxes=ground_truth_bboxes, iou_threshold=50/100))

    return sum(iou_thresholds_mean_average_precision)/len(iou_thresholds_mean_average_precision)


def modified_sigmoid(x, coefficient=0.5):
    return 1 / (1 + torch.exp(-coefficient * x))