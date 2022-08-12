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
		return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(num_input_params, 512), torch.nn.Dropout(DROPOUT),
		                           torch.nn.LeakyReLU(0.1), torch.nn.Linear(512, grid_size_width * grid_size_height * (number_classes + 5)))
