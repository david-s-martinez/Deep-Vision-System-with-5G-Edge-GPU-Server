import torch
import pandas as pd
import numpy as np
import cv2


class DetectionDataset(torch.utils.data.Dataset):
	def __init__(self, csv_file, grid_size_width, grid_size_height, number_classes, anchor_boxes, transformations=None):
		self.images_labels_paths_dataframe = pd.read_csv(csv_file)
		self.grid_size_width = grid_size_width
		self.grid_size_height = grid_size_height
		self.number_classes = number_classes
		self.transformations = transformations
		self.anchor_boxes = torch.tensor(anchor_boxes)
		self.num_anchors = self.anchor_boxes.shape[0]

	def __len__(self):
		return len(self.images_labels_paths_dataframe)

	def __getitem__(self, index):
		image_path = self.images_labels_paths_dataframe.iloc[index, 0]
		label_path = self.images_labels_paths_dataframe.iloc[index, 1]

		image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
		ground_truth_bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()

		if self.transformations:
			augmentations = self.transformations(image=image, bboxes=ground_truth_bboxes)
			image = augmentations['image']
			ground_truth_bboxes = augmentations['bboxes']
		label = torch.zeros((self.grid_size_height, self.grid_size_width, 8))

		for bbox in ground_truth_bboxes:
			x_relative_image, y_relative_image, w_relative_image, h_relative_image, class_label = bbox
			cell_row, cell_column = int(self.grid_size_height * y_relative_image), int(self.grid_size_width * x_relative_image)

			x_relative_cell, y_relative_cell = x_relative_image * self.grid_size_width - cell_column, y_relative_image * self.grid_size_height - cell_row
			w_relative_cell, h_relative_cell = w_relative_image * self.grid_size_width, h_relative_image * self.grid_size_height

			if label[cell_row, cell_column, self.number_classes] == 0:
				label[cell_row, cell_column, self.number_classes] = 1
				label[cell_row, cell_column, int(class_label)] = 1
				label[cell_row, cell_column, self.number_classes + 1:] = torch.tensor([x_relative_cell, y_relative_cell, w_relative_cell, h_relative_cell])

		return image, label
