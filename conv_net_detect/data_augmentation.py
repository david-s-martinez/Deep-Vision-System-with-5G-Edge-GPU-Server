import albumentations as A
import cv2
import matplotlib.pyplot as plt
from CONFIG import IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT

from albumentations.pytorch import ToTensorV2


transformations_detection = A.Compose(
	[
		A.Sequential([
			A.Rotate(limit=90, p=0.5, border_mode=0, method="ellipse"),
			A.Resize(width=IMAGE_SIZE_WIDTH, height=IMAGE_SIZE_HEIGHT, p=1),
		]),

		A.Sequential([
			A.RandomCrop(height=170, width=250, p=0.5),
			A.Resize(width=IMAGE_SIZE_WIDTH, height=IMAGE_SIZE_HEIGHT, p=1),
		]),
		A.HorizontalFlip(p=0.5),
		A.VerticalFlip(p=0.5),
		A.OneOf([
			A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.5),
			A.ISONoise(color_shift=(0.01, 1), intensity=(0.01, 1), p=0.5),
			A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=True, elementwise=True, p=0.5),

		], p=0.5),

		A.OneOf([
			A.RandomBrightness(p=0.5, limit=(-0.3, 0.3)),
			A.RandomContrast(p=0.5, limit=(-0.4, 0.5))
		], p=0.5),
		A.OneOf([
			A.Blur(p=0.5, blur_limit=3),
			A.GaussianBlur(p=0.5, blur_limit=(1, 3), sigma_limit=0),
			A.MotionBlur(p=0.75, blur_limit=5)
		], p=0.5),
		A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
		ToTensorV2()
	], bbox_params=A.BboxParams(format="yolo", min_visibility=0.3)
)




