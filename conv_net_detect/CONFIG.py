import torch

CONFIDENCE_THRESHOLD = 0.75
IOU_THRESHOLD = 0.3
NUMBER_CLASSES = 3
GRID_SIZE_WIDTH = 18
GRID_SIZE_HEIGHT = 12
LEARNING_RATE = 1.2 * pow(10, -4)
BATCH_SIZE_TESTING = 1
DATASET_SPLIT = (1, 1)
NUMBER_EPOCHS = 500

CHOSEN_MODEL = 2

if CHOSEN_MODEL == 0:
	MODEL_NAME = "RESNET_18"
	WEIGHT_DECAY = 0.0002
	DROPOUT = 0.
	BATCH_SIZE_TRAINING = 256
elif CHOSEN_MODEL == 1:
	MODEL_NAME = "RESNET_50"
	WEIGHT_DECAY = 0.0002
	DROPOUT = 0.2
	BATCH_SIZE_TRAINING = 32
elif CHOSEN_MODEL == 2:
	MODEL_NAME = "MOBILENET_V2"
	WEIGHT_DECAY = 0.0000
	DROPOUT = 0.0
	BATCH_SIZE_TRAINING = 128

CHOOSE_CUDA = 6
DEVICE = torch.device(CHOOSE_CUDA if torch.cuda.is_available() else "cpu")

CSV_FILE_CLASSIFICATION = "classification_dataset_description.csv"
CSV_FILE_DETECTION_TRAINING = "detection_subset_training.csv"
CSV_FILE_DETECTION_TESTING = "detection_subset_testing.csv"

#ROOT_PATH = "/home.stud/morozart/DeltaWheelsDetection/"
ROOT_PATH = "/Users/artemmoroz/Desktop/CIIRC_projects/Detection_artificial_dataset_v3/"
ROOT_PATH_COLAB = "/content/gdrive/MyDrive/delta_detection_v1/DeltaWheelsDetection2/"

IMAGE_SIZE_WIDTH = 270
IMAGE_SIZE_HEIGHT = 180
ANCHOR_BOXES = [[3.5] * 2] #torch.tensor([[4.2] * 2, [3.5] * 2, [2.9] * 2]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
#ANCHOR_BOXES = [[2.93333333] * 2, [3.13333333] * 2, [3.33333333] * 2, [3.46666667] * 2,  [3.66666667] * 2,  [3.8] * 2, [3.93333333] * 2, [4.13333333] * 2, [4.33333333] * 2, [4.53333333] * 2]

