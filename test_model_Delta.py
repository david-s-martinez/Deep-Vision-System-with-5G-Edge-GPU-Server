import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import Detection_models
ANCHOR_BOXES = [3.8] * 2
GRID_SIZE_WIDTH = 18
GRID_SIZE_HEIGHT = 12
# DEVICE = 0 if torch.cuda.is_available() else "cpu"
NUMBER_CLASSES = 3
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.1


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
            if not is_predictions and not torch.all(bboxes_reshaped[image_index, cell_index, :].eq(torch.zeros(6).to(0))):
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
    bboxes[..., :1] = (bboxes[..., :1] + cell_indicis_x.to(0))/GRID_SIZE_WIDTH
    bboxes[..., 1:2] = (bboxes[..., 1:2] + cell_indicis_y.to(0))/GRID_SIZE_HEIGHT
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


def get_bboxes(predicted_bboxes, image_index):

    bboxes_predicted_relative_image = convert_cell_to_image(predicted_bboxes[..., 4:])
    predicted_class = predicted_bboxes[..., :NUMBER_CLASSES].argmax(-1).unsqueeze(-1)
    predicted_bboxes_modified = torch.cat([predicted_class, predicted_bboxes[..., NUMBER_CLASSES:NUMBER_CLASSES + 1], bboxes_predicted_relative_image], dim=-1)
    predicted_bboxes_list = get_bboxes_list(predicted_bboxes_modified, is_predictions=True, index=image_index)
    predicted_bbox_after_non_max_suppression = non_maxima_suppression(predicted_bboxes_list)

    return predicted_bbox_after_non_max_suppression


def make_prediction(frame,model):
    
    predictions = model(frame)

    predictions[..., 4:6] = modified_sigmoid(predictions[..., 4:6], coefficient=0.75)
    predictions[..., 6:] = torch.tensor(ANCHOR_BOXES).reshape(1, 1, 1, 2).to(0) * modified_sigmoid(predictions[..., 6:], coefficient=0.75)
    bboxes = get_bboxes(predicted_bboxes=predictions, image_index=0)[0]
    bboxes_to_return = list()
    for bbox in bboxes:
        x_min = bbox[3] - bbox[5]/2
        y_min = bbox[4] - bbox[6]/2
        x_max = bbox[3] + bbox[5]/2
        y_max = bbox[4] + bbox[6]/2
        confidence_score = bbox[2] if bbox[2] < 1.0 else 1.0
        predicted_class = bbox[1]
        bboxes_to_return.append([x_min, y_min, x_max, y_max, confidence_score, predicted_class])
    return bboxes_to_return

def detection(frame, model, show_centroids=True):
    frame = reshape(frame, w=270, h=180)
    frame_normalized = image_normalization(image=frame).unsqueeze(0).float().to(0)
    bboxes = make_prediction(frame_normalized, model)
    return bboxes