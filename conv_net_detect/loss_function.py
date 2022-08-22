import torch
import utils
from CONFIG import DEVICE


class LossFunction(torch.nn.Module):
	def __init__(self, grid_size, number_classes, anchor_boxes):
		super(LossFunction, self).__init__()

		self.grid_size = grid_size
		self.number_classes = number_classes
		self.MSE = torch.nn.MSELoss()
		self.BCE = torch.nn.BCELoss()
		self.CE = torch.nn.CrossEntropyLoss()
		self.anchor_boxes = anchor_boxes
		self.lambda_bbox_loss = 5
		self.lambda_class_loss = 3
		self.lambda_no_objet_loss = 2

	def forward(self, predictions, labels):
		#   prediction.shape = [Batch_size, Grid_size, Grid_size, 8] where the last dimension is [p_c1, p_c2, p_c3, confidence, x, y, w, h]
		#   labels.shape = [Batch_size, Grid_size, Grid_size, 8] where the last dimension is [p_c1, p_c2, p_c3, confidence, x, y, w, h]
		predictions[..., 4:6] = utils.modified_sigmoid(predictions[..., 4:6], coefficient=0.5)
		predictions[..., 6:] = torch.tensor(self.anchor_boxes).reshape(1, 1, 2 * len(self.anchor_boxes)).to(DEVICE) * utils.modified_sigmoid(predictions[..., 6:], coefficient=0.5)

		object_mask = labels[..., self.number_classes] == 1
		no_object_mask = labels[..., self.number_classes] == 0

		no_object_loss = self.MSE(torch.flatten(predictions[..., self.number_classes:self.number_classes + 1][no_object_mask]),
		                          torch.flatten(labels[..., self.number_classes:self.number_classes + 1][no_object_mask]))

		object_loss = self.MSE(torch.flatten(predictions[..., self.number_classes:self.number_classes + 1][object_mask]),
		                       torch.flatten(labels[..., self.number_classes:self.number_classes + 1][object_mask]))

		bbox_loss1 = self.MSE(torch.flatten(predictions[..., -4:-2][object_mask], end_dim=-2), torch.flatten(labels[..., -4:-2][object_mask], end_dim=-2))
		bbox_loss2 = self.MSE(torch.flatten(predictions[..., -2:][object_mask], end_dim=-2), torch.flatten((labels[..., -2:][object_mask]), end_dim=-2))

		class_loss = self.MSE(torch.flatten(predictions[..., :self.number_classes][object_mask], end_dim=-2), torch.flatten(labels[..., :self.number_classes][object_mask], end_dim=-2))

		return object_loss + self.lambda_no_objet_loss * no_object_loss + self.lambda_bbox_loss * (bbox_loss1 + bbox_loss2) + self.lambda_class_loss * class_loss

