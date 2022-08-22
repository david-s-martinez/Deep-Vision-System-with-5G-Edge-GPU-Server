import random
import copy
#from Detection_artificial_dataset import reshape, check_collision
from utils import rotate, remove_background, remove_white, remove_black, get_gripper_mask, compute_IOU
import numpy as np
import time
import cv2
import torch
import sys
import os
import csv
import matplotlib.pyplot as plt
from CONFIG import ROOT_PATH, DATASET_SPLIT, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, ROOT_PATH_COLAB
import albumentations as A


def reshape(image, w, h):
	"""
	Function reshapes a batch of images into 256x256 size
	:param images:
	:type ndarray:
	:return reshaped:
	:type ndarray:
	"""

	reshaped_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)

	return reshaped_image


def check_collision(data_dict, x_middle1, y_middle1, w1, h1):
	for key in list(data_dict.keys()):
		x_middle2, y_middle2, w2, h2 = data_dict[key][0], data_dict[key][1], data_dict[key][2] + 25, data_dict[key][2] + 25

		IOU = compute_IOU(torch.tensor([x_middle1, y_middle1, w1, h1]), torch.tensor([x_middle2, y_middle2, w2, h2]))
		if IOU.item() > 0.0:
			return False
	return True


def compute_distances(camera_point, x_y_middle_list):
	distances = list()

	for x_y_middle in x_y_middle_list:
		distances.append(compute_distance(camera_point, x_y_middle))

	return distances


def compute_distance(camera_point, x_y_middle):

	return (camera_point[0] - x_y_middle[0])**2 + int(1.25 * (camera_point[1] - x_y_middle[1]))**2


def insert_objects(image_base, camera_point, object_list, delta_list, x_y_middle_list):
	distances = compute_distances(camera_point, x_y_middle_list)
	distances_ordered = sorted(distances, reverse=True)
	number_objects = len(distances)

	for object_number in range(number_objects):

		if len(distances_ordered) == 0:
			break
		max_dist_index = distances.index(distances_ordered[0])
		distances_ordered.pop(0)
		rotated_wheel = object_list[max_dist_index]
		x_middle, y_middle = x_y_middle_list[max_dist_index]
		delta_h, delta_w = delta_list[max_dist_index]
		if delta_h == 0:
			image_base = remove_redundant_white(image_base, insertion=rotated_wheel[...], x_middle=x_middle, y_middle=y_middle,
			                                    height=int(rotated_wheel.shape[0] / 2) - delta_h, width=int(rotated_wheel.shape[1] / 2) - delta_w)
		else:
			image_base = remove_redundant_white(image_base, insertion=rotated_wheel[delta_h:-delta_h, delta_w:-delta_w, :], x_middle=x_middle, y_middle=y_middle,
			                                    height=int(rotated_wheel.shape[0] / 2) - delta_h, width=int(rotated_wheel.shape[1] / 2) - delta_w)

	return image_base


def add_gripper(image, gripper):
	gripper_visibility = random.randint(1, 474)
	gripper_side = random.randint(0, 3)
	gripper = rotate(image=gripper, angle=-90 * gripper_side)
	if gripper_side in [0, 2]:
		gripper_point_place = random.randint(0, 1500)
		if gripper_side == 0:
			gripper_image = gripper[474 - gripper_visibility:, :1599 - gripper_point_place if 1599 - gripper_point_place < 755 else 755]
			gripper_mask = get_gripper_mask(gripper_image)
			image[:gripper_visibility, gripper_point_place:(gripper_point_place + 755) if (gripper_point_place + 755) < 1599 else 1599] = \
				image[:gripper_visibility, gripper_point_place:(gripper_point_place + 755) if (gripper_point_place + 755) < 1599 else 1599] * \
				(1 - np.transpose(np.repeat(gripper_mask, [3], axis=0), axes=[1, 2, 0])) + \
				gripper_image * np.transpose(np.repeat(gripper_mask, [3], axis=0), axes=[1, 2, 0])

		else:
			gripper_image = gripper[:gripper_visibility, :1599 - gripper_point_place if 1599 - gripper_point_place < 755 else 755]
			gripper_mask = get_gripper_mask(gripper_image)
			image[-gripper_visibility:, gripper_point_place:(gripper_point_place + 755) if (gripper_point_place + 755) < 1599 else 1599] = \
				image[-gripper_visibility:, gripper_point_place:(gripper_point_place + 755) if (gripper_point_place + 755) < 1599 else 1599] * \
				(1 - np.transpose(np.repeat(gripper_mask, [3], axis=0), axes=[1, 2, 0])) + \
				gripper_image * np.transpose(np.repeat(gripper_mask, [3], axis=0), axes=[1, 2, 0])

	else:
		gripper_point_place = random.randint(0, 900)
		if gripper_side == 1:
			gripper_image = gripper[:999 - gripper_point_place if 999 - gripper_point_place < 755 else 755, :gripper_visibility]
			gripper_mask = get_gripper_mask(gripper_image)
			image[gripper_point_place:(gripper_point_place + 755) if (gripper_point_place + 755) < 999 else 999, -gripper_visibility:] = \
				image[gripper_point_place:(gripper_point_place + 755) if (gripper_point_place + 755) < 999 else 999, -gripper_visibility:] * \
				(1 - np.transpose(np.repeat(gripper_mask, [3], axis=0), axes=[1, 2, 0])) + \
				gripper_image * np.transpose(np.repeat(gripper_mask, [3], axis=0), axes=[1, 2, 0])

		else:
			gripper_image = gripper[:999 - gripper_point_place if 999 - gripper_point_place < 755 else 755, 474 - gripper_visibility:]
			gripper_mask = get_gripper_mask(gripper_image)
			image[gripper_point_place:(gripper_point_place + 755) if (gripper_point_place + 755) < 999 else 999, :gripper_visibility] = \
				image[gripper_point_place:(gripper_point_place + 755) if (gripper_point_place + 755) < 999 else 999, :gripper_visibility] * \
				(1 - np.transpose(np.repeat(gripper_mask, [3], axis=0), axes=[1, 2, 0])) + \
				gripper_image * np.transpose(np.repeat(gripper_mask, [3], axis=0), axes=[1, 2, 0])

	return image


def get_all_objects_mask(image):

	zeros = 150 * np.ones((image.shape[0], image.shape[1]))
	R_channel = image[..., 0] <= zeros
	G_channel = image[..., 1] <= zeros
	B_channel = image[..., 2] <= zeros
	object_mask = np.array(([R_channel * G_channel * B_channel]))
	return object_mask


def add_object_shadows(image, data_dict):
	object_mask = get_no_objects_mask(image)
	image_no_objects = image * (np.transpose(np.repeat(object_mask, [3], axis=0), axes=[1, 2, 0]))

	shadow_color_base = np.array([145, 143, 138]) + np.random.uniform(low=-15, high=15)
	shadow_random_color_shift = np.random.uniform(low=-5, high=5)
	image_objects = (image[..., 0:3] - shadow_color_base - shadow_random_color_shift) * (1 - np.transpose(np.repeat(object_mask, [3], axis=0), axes=[1, 2, 0]))
	shadow_color = (shadow_color_base[0] + shadow_random_color_shift, shadow_color_base[1] + shadow_random_color_shift, shadow_color_base[2] + shadow_random_color_shift)

	shadow_random_height_shift_1 = random.randint(-5, 5)
	shadow_random_height_shift_2 = random.randint(-4, 4)
	shadow_random_width_shift_1 = random.randint(-5, 5)
	shadow_random_width_shift_2 = random.randint(-4, 4)

	for key, value in data_dict.items():
		x_coord, y_coord = int(value[0] / (IMAGE_SIZE_WIDTH/1600)), int(value[1] / (IMAGE_SIZE_WIDTH/1600)) + 50
		magnitude = value[-2]
		angle_wheel_rotation = value[-1]

		if value[-3] in [0, 2]:
			if angle_wheel_rotation > 0:
				left_shift = -int(np.abs(angle_wheel_rotation) * 0.4) - int(1.25 * value[2])
				right_shift = int(np.abs(angle_wheel_rotation) * 1.2) + int(1.25 * value[2])
			else:
				left_shift = -int(np.abs(angle_wheel_rotation) * 1.2) - int(1.25 * value[2])
				right_shift = int(np.abs(angle_wheel_rotation) * 0.4) + int(1.25 * value[2])

			left_shift += shadow_random_width_shift_1
			right_shift -= shadow_random_width_shift_1

			height_up_shift = int(value[2] * 1 + 0.4 * max(np.abs(angle_wheel_rotation), np.abs(65 - angle_wheel_rotation))) - shadow_random_height_shift_1
			height_down_shift = int(magnitude * 25) + shadow_random_height_shift_1
			radius = int(value[2] * 2.25)
			upper_radius = radius + 50

		else:
			if angle_wheel_rotation > 0:
				left_shift = -int(np.abs(angle_wheel_rotation) * 0.3) - int(1.1 * value[2])
				right_shift = int(np.abs(angle_wheel_rotation) * 1.25) + int(1.1 * value[2])
			else:
				left_shift = -int(np.abs(angle_wheel_rotation) * 1.25) - int(1.1 * value[2])
				right_shift = int(np.abs(angle_wheel_rotation) * 0.3) + int(1.1 * value[2])

			left_shift += shadow_random_width_shift_2
			right_shift -= shadow_random_width_shift_2

			height_up_shift = int(value[2] * 1 + 0.55 * max(np.abs(angle_wheel_rotation), np.abs(65 - angle_wheel_rotation))) - shadow_random_height_shift_2
			height_down_shift = int(magnitude * 16) + shadow_random_height_shift_2
			radius = int(value[2] * 2.25)
			upper_radius = radius + 40

		image_no_objects = cv2.circle(image_no_objects, center=(x_coord, y_coord - height_up_shift), radius=upper_radius, color=shadow_color, thickness=-1)
		image_no_objects = cv2.circle(image_no_objects, center=(x_coord + int(1.5 * angle_wheel_rotation), y_coord + height_down_shift), radius=radius, color=shadow_color, thickness=-1)
		image_no_objects = cv2.circle(image_no_objects, center=(x_coord + left_shift, y_coord - int(np.abs(angle_wheel_rotation) * 0.75)), radius=radius, color=shadow_color, thickness=-1)
		image_no_objects = cv2.circle(image_no_objects, center=(x_coord + right_shift, y_coord - int(np.abs(angle_wheel_rotation) * 0.75)), radius=radius, color=shadow_color, thickness=-1)

	for _ in range(5):
		image_no_objects = apply_blur(image_no_objects)
	image = image_no_objects + image_objects
	#plt.imshow(image_no_objects/255)
	#plt.show()
	#exit()
	return image


def apply_blur(image, blur_limit=99):
	blur_transform = A.Compose([
		A.GaussianBlur(p=1, blur_limit=(blur_limit, blur_limit + 2), sigma_limit=[30, 31], always_apply=True)
	])

	transformed_image = blur_transform(image=image)["image"]

	return transformed_image


def add_margin_image(image, border_width, margin_thickness):
	image = np.zeros(image.shape)
	x_min, y_min = border_width - margin_thickness, border_width - margin_thickness
	bottom_aruco_height = border_width - margin_thickness - 100

	main_margin_color = np.array([166 - 194 - 5, 164 - 197, 163 - 197]) + np.random.randint(low=-10, high=10)
	random_color_shift = 15
	shadowed_margin_color = [118 - 194 - 90, 125 - 197 - 90, 128 - 197 - 90]

	shadowed_margin_thickness = 3

	image[y_min:image.shape[0] - bottom_aruco_height, x_min:x_min + margin_thickness - shadowed_margin_thickness, :] = \
		np.array(main_margin_color) + np.random.uniform(low=-random_color_shift, high=random_color_shift, size=(image.shape[0] - y_min - bottom_aruco_height, margin_thickness -
		                                                                                                       shadowed_margin_thickness, 3)) #* 10

	image[y_min:y_min + margin_thickness - shadowed_margin_thickness, x_min:image.shape[1] - x_min, :] = \
		np.array(main_margin_color) + np.random.uniform(low=-random_color_shift, high=random_color_shift, size=(margin_thickness - shadowed_margin_thickness, image.shape[1] - 2 * y_min, 3)) #* 10

	image[y_min:image.shape[0] - bottom_aruco_height, image.shape[1] - x_min - margin_thickness + shadowed_margin_thickness:image.shape[1] - x_min, :] = \
		np.array(main_margin_color) + np.random.uniform(low=-random_color_shift, high=random_color_shift, size=(image.shape[0] - y_min - bottom_aruco_height, margin_thickness -
		                                                                                                        shadowed_margin_thickness, 3)) #* 10

	"""image[image.shape[0] - bottom_aruco_height - margin_thickness + shadowed_margin_thickness:image.shape[0] - bottom_aruco_height, x_min:image.shape[1] - x_min, :] = \
		np.array(main_margin_color) + np.random.uniform(low=-random_color_shift, high=random_color_shift, size=(margin_thickness - shadowed_margin_thickness, image.shape[1] - 2 * x_min, 3))"""# * 10

	image[image.shape[0] - bottom_aruco_height - margin_thickness:image.shape[0] - bottom_aruco_height, x_min:image.shape[1] - x_min, :] = \
		np.array(main_margin_color) + np.random.uniform(low=-random_color_shift, high=random_color_shift, size=(margin_thickness, image.shape[1] - 2 * x_min, 3))


	image[border_width:image.shape[0] - bottom_aruco_height - margin_thickness, x_min + margin_thickness - shadowed_margin_thickness:x_min + margin_thickness, :] = \
		np.array(shadowed_margin_color) + np.random.uniform(low=-random_color_shift + 5, high=random_color_shift - 5, size=(image.shape[0] - bottom_aruco_height - margin_thickness - border_width,
		                                                                                                       shadowed_margin_thickness,
		                                                                                       3))

	image[y_min + margin_thickness - shadowed_margin_thickness:y_min + margin_thickness, border_width:image.shape[1] - border_width, :] = \
		np.array(shadowed_margin_color) + np.random.uniform(low=-random_color_shift + 5, high=random_color_shift - 5, size=(shadowed_margin_thickness, image.shape[1] - 2 * border_width, 3))

	image[border_width:image.shape[0] - bottom_aruco_height - margin_thickness, image.shape[1] - x_min - margin_thickness:image.shape[1] - x_min - margin_thickness + shadowed_margin_thickness, :] = \
		np.array(shadowed_margin_color) + np.random.uniform(low=-random_color_shift + 5, high=random_color_shift - 5, size=(image.shape[0] - margin_thickness - bottom_aruco_height - border_width,
		                                                                                                       shadowed_margin_thickness,
		                                                                                       3))

	#image[image.shape[0] - bottom_aruco_height - margin_thickness:image.shape[0] - bottom_aruco_height - margin_thickness + shadowed_margin_thickness, border_width:image.shape[1] - border_width,
	# :] = \
		#np.array(shadowed_margin_color) + np.random.uniform(low=-random_color_shift + 5, high=random_color_shift - 5, size=(shadowed_margin_thickness, image.shape[1] - 2 * border_width, 3))

	image = apply_blur(image=image, blur_limit=11)
	return image


def add_aruco_tags(image, border_width, margin_thickness, aruco_tags_images):
	aruco_tags = list()
	aruco_tag_height = list()

	image = np.zeros(image.shape)
	for aruco_tag_index in range(10):
		aruco_tag = aruco_tags_images[aruco_tag_index]
		aruco_width = border_width - margin_thickness
		if aruco_tag_index + 1 in [6, 7, 8, 9]:
			aruco_height = border_width - margin_thickness - 100
			aruco_tag = reshape(aruco_tag, w=aruco_width, h=aruco_height)
			aruco_tag_height.append(aruco_height)
		else:
			aruco_height = border_width - margin_thickness
			aruco_tag = reshape(aruco_tag, w=aruco_width, h=aruco_height)
			aruco_tag_height.append(aruco_height)
		aruco_tags.append(aruco_tag)
	aruco_tag_coords = [(0, 0), (500, 0), (1000, 0), (1600 - aruco_tag_height[3], 0), (1600 - aruco_tag_height[4], 434),
	                    (1600 - aruco_tag_height[4], 1000 - aruco_tag_height[5]), (1000, 1000 - aruco_tag_height[6]), (500, 1000 - aruco_tag_height[7]), (0, 1000 - aruco_tag_height[8]), (0, 434)]

	for index in range(len(aruco_tag_coords)):
		image[aruco_tag_coords[index][1]:aruco_tag_coords[index][1] + aruco_tag_height[index], aruco_tag_coords[index][0]:aruco_tag_coords[index][0] + aruco_tag_height[0]] = (aruco_tags[index] -
		                                                                                                                                                                       np.array([205]))

	return image


def remove_background_selected_area(image, x_middle, y_middle, height, width):
	ones = np.ones((image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0].shape[0], image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0].shape[1]))

	R_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0] <= 240 * ones
	G_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 1] <= 240 * ones
	B_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 2] <= 240 * ones

	mask = np.array([R_channel * G_channel * B_channel])
	image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, :] = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, :] * (np.transpose(np.repeat(mask, [3], axis=0), axes=[1, 2, 0]))
	return image


def remove_redundant_white(image, insertion, x_middle, y_middle, height, width):

	if insertion.shape[1] != (x_middle + width) - (x_middle - width) or insertion.shape[0] != (y_middle + height) - (y_middle - height):
		insertion = reshape(insertion, h=(y_middle + height) - (y_middle - height), w=(x_middle + width) - (x_middle - width))

	image = copy.deepcopy(image)
	ones = np.ones(
		(image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0].shape[0], image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0].shape[1]))

	R_channel_ = insertion[..., 0] < 150 * ones
	G_channel_ = insertion[..., 1] < 150 * ones
	B_channel_ = insertion[..., 2] < 150 * ones

	object_part_mask_new = np.array([R_channel_ * G_channel_ * B_channel_])

	R_channel_ = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0] < 150 * ones
	G_channel_ = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 1] < 150 * ones
	B_channel_ = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 2] < 150 * ones

	object_part_mask_old = np.array([R_channel_ * G_channel_ * B_channel_])

	image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, :] -= \
	image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, :] * \
	np.transpose(np.repeat(object_part_mask_new * object_part_mask_old, [3], axis=0), axes=[1, 2, 0])
	image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, :] += \
		insertion

	R_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0] >= 255 * ones
	G_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 1] >= 255 * ones
	B_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 2] >= 255 * ones

	mask = np.array([R_channel * G_channel * B_channel])

	image[y_middle - height:y_middle + height, x_middle - width:x_middle + width, :] = \
		image[y_middle - height:y_middle + height, x_middle - width:x_middle + width, :] - \
		np.array([255]) * np.transpose(np.repeat(mask, [3], axis=0), axes=[1, 2, 0])

	return image


def plot_bbox(image, x_middle, y_middle, w, h, wheel_type):
	cv2.rectangle(image, (int(x_middle - w / 2), int(y_middle - h / 2)), (int(x_middle + w / 2), int(y_middle + h / 2)), (255, 0, 0), 1)
	cv2.putText(image, str(wheel_type), (x_middle, y_middle-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

	return image


def get_no_objects_mask(image):
	threshold = np.ones((image.shape[0], image.shape[1])) * (170)
	R_channel = image[..., 0] >= threshold
	G_channel = image[..., 1] >= threshold
	B_channel = image[..., 2] >= threshold
	no_object_mask = np.array(([R_channel * G_channel * B_channel]))

	return no_object_mask


def add_background_shadows(image):
	no_object_mask = get_no_objects_mask(image)
	random_gray_color_shift = np.random.uniform(low=-1, high=1, size=(image.shape[0], image.shape[1], 1))

	background = (image + np.repeat(random_gray_color_shift, [3], axis=-1)) * np.transpose(np.repeat(no_object_mask, [3], axis=0), axes=[1, 2, 0])

	full_image = background + image * (1 - np.transpose(np.repeat(no_object_mask, [3], axis=0), axes=[1, 2, 0]))
	return full_image


def generate_image(objects_transformed, aruco_tags, gripper):
	background_height, background_width = 1000, 1600
	image_base = np.ones((background_height, background_width, 3)) * 255
	max_wheels = [5, 5, 4, 4, 3, 3]

	distance_array = [1, 1.05, 1.1, 1.15, 1.2, 1.25]
	distance_index = random.randint(5, 5)
	distance = distance_array[distance_index]
	wheel_width = [314, 335, 349, 360, 374, 390][distance_index]
	wheel_height_magnitude_1, wheel_height_magnitude_2, wheel_height_magnitude_3 = [304, 320, 334, 350, 364, 380][distance_index], \
	                                                                               [332, 348, 366, 382, 398, 414][distance_index], \
	                                                                               [360, 388, 407, 424, 445, 460][distance_index]

	disk_width = [258, 272, 284, 298, 314, 323][distance_index]
	disk_height_magnitude_1, disk_height_magnitude_2, disk_height_magnitude_3 = [257, 273, 290, 302, 314, 330][distance_index], \
	                                                                            [282, 296, 311, 325, 338, 352][distance_index], \
	                                                                            [290, 308, 320, 336, 352, 373][distance_index]

	border_width = int(138 * distance)
	wheel_bbox_size = [52, 56, 59, 62, 65, 68][distance_index]
	disk_bbox_size = int(wheel_bbox_size * 0.85)
	margin_thickness = [16, 17, 18, 19, 20, 21][distance_index] + 4
	camera_central_point = (random.randint(0, background_width), random.randint(700, background_height))
	#angle_camera_perspective = (-60 / (background_width / 2)) * (camera_central_point[0] - background_width / 2)
	#top_black_point = 0.5 * (camera_central_point[0] + 2400)
	#bottom_black_point = top_black_point - 1000 * np.tan(angle_camera_perspective) if top_black_point >= 800 else top_black_point + 1000 * np.tan(angle_camera_perspective)

	disk_magnitude_1 = reshape(objects_transformed[0], w=disk_width, h=disk_height_magnitude_1)
	disk_magnitude_2 = reshape(objects_transformed[1], w=disk_width, h=disk_height_magnitude_2)
	disk_magnitude_3 = reshape(objects_transformed[2], w=disk_width, h=disk_height_magnitude_3)
	disk_dark_magnitude_1 = reshape(objects_transformed[3], w=disk_width, h=disk_height_magnitude_1)
	disk_dark_magnitude_2 = reshape(objects_transformed[4], w=disk_width, h=disk_height_magnitude_2)
	disk_dark_magnitude_3 = reshape(objects_transformed[5], w=disk_width, h=disk_height_magnitude_3)
	tyre_magnitude_1 = reshape(objects_transformed[6], w=wheel_width, h=wheel_height_magnitude_1)
	tyre_magnitude_2 = reshape(objects_transformed[7], w=wheel_width, h=wheel_height_magnitude_2)
	tyre_magnitude_3 = reshape(objects_transformed[8], w=wheel_width, h=wheel_height_magnitude_3)
	wheel_magnitude_1 = reshape(objects_transformed[9], w=wheel_width, h=wheel_height_magnitude_1)
	wheel_magnitude_2 = reshape(objects_transformed[10], w=wheel_width, h=wheel_height_magnitude_2)
	wheel_magnitude_3 = reshape(objects_transformed[11], w=wheel_width, h=wheel_height_magnitude_3)
	wheel_dark_magnitude_1 = reshape(objects_transformed[12], w=wheel_width, h=wheel_height_magnitude_1)
	wheel_dark_magnitude_2 = reshape(objects_transformed[13], w=wheel_width, h=wheel_height_magnitude_2)
	wheel_dark_magnitude_3 = reshape(objects_transformed[14], w=wheel_width, h=wheel_height_magnitude_3)

	images_dict = {"0": [tyre_magnitude_1, tyre_magnitude_2, tyre_magnitude_3],
	               "1": [disk_magnitude_1, disk_magnitude_2, disk_magnitude_3],
	               "2": [disk_dark_magnitude_1, disk_dark_magnitude_2, disk_dark_magnitude_3],
	               "3": [wheel_magnitude_1, wheel_magnitude_2, wheel_magnitude_3],
	               "4": [wheel_dark_magnitude_1, wheel_dark_magnitude_2, wheel_dark_magnitude_3]}

	number_wheels = random.randint(3, max_wheels[distance_index] + 2)
	print("Number of wheels {}".format(number_wheels))
	data_dict = dict()

	objects_list = list()
	delta_list = list()
	x_y_middle_list = list()

	for i in range(number_wheels):
		wheel_type = random.randint(0, 4)
		wheels_set = images_dict[str(wheel_type)]
		time1 = time.time()
		while True:
			if time.time() - time1 > 2.5:
				print("TIME EXCEEDED!")
				print(time.time() - time1)
				break
			y_upper_margin, y_lower_margin = border_width + int(wheel_height_magnitude_3/2), background_height - 30 - int(wheel_height_magnitude_3/2)
			x_middle = random.randint(border_width + int(wheel_width / 2), background_width - border_width - int(wheel_width / 2))
			y_middle = random.randint(y_upper_margin, y_lower_margin)
			
			magnitude = 1
			if camera_central_point[1] - y_middle <= 287 - int(wheel_height_magnitude_3/2):
				magnitude = 1
			elif 287 - int(wheel_height_magnitude_3/2) <= camera_central_point[1] - y_middle <= 575 - int(wheel_height_magnitude_3/2):
				magnitude = 2
			elif 575 - int(wheel_height_magnitude_3/2) <= camera_central_point[1] - y_middle <= 862 - int(wheel_height_magnitude_3/2):
				magnitude = 3
			angle_rotation = (60 / 1186) * (- x_middle + camera_central_point[0])

			rotated_wheel = rotate(wheels_set[magnitude - 1], angle_rotation)
			height_min = wheels_set[magnitude - 1].shape[0] / 2
			width_min = wheels_set[magnitude - 1].shape[1] / 2
			height_width_max = np.sqrt(height_min ** 2 + width_min ** 2)

			if wheel_type == 0:
				wheel_class = 0
				bbox_size = wheel_bbox_size
			elif wheel_type in [1, 2]:
				wheel_class = 1
				bbox_size = disk_bbox_size
			elif wheel_type in [3, 4]:
				wheel_class = 2
				bbox_size = wheel_bbox_size

			if wheel_class in [0, 2]:
				angle_diff = 0
				if 0 <= np.abs(angle_rotation) <= 30:
					angle_diff = 45
				delta_h = int((height_width_max - height_min) * np.sin(np.deg2rad(np.abs(angle_rotation) + 50 - angle_diff)))
				delta_w = int((height_width_max - height_min) * np.sin(np.deg2rad(np.abs(angle_rotation) + 50 - angle_diff)))

			else:
				angle_diff = 0
				if 0 <= np.abs(angle_rotation) <= 30:
					angle_diff = 35
				delta_h = int((height_width_max - height_min) * np.sin(np.deg2rad(np.abs(angle_rotation) + 40 - angle_diff)))
				delta_w = int((height_width_max - height_min) * np.sin(np.deg2rad(np.abs(angle_rotation) + 40 - angle_diff)))

			if angle_rotation > 0:
				if wheel_class == 1:
					if magnitude == 1:
						height_shift = int(15 / distance) + int(angle_rotation * 0.35) - 0.33 * (60 - angle_rotation) * np.tanh((350 - y_middle) * 0.01)
						width_shift = -angle_rotation * 0.5 / distance + 8
					elif magnitude == 2:
						height_shift = int(15 / distance) + int(angle_rotation * 0.15) - 0.2 * (60 - angle_rotation) * np.tanh((500 - y_middle) * 0.01)
						width_shift = - angle_rotation * 0.45 / distance + 8
					else:
						height_shift = int(1 / distance) + int(angle_rotation * 0.1) - 0.33 * (60 - angle_rotation) * np.tanh((350 - y_middle) * 0.01)
						width_shift = -angle_rotation * 0.6 / distance + 8
					height_shift -= np.clip(np.abs(45 - angle_rotation), 0, 45) * magnitude * 0.1
				else:
					if magnitude == 1:
						if y_middle > (y_lower_margin - y_upper_margin)/2:
							height_shift = 19 * distance * np.exp(0.0015 * (y_middle - (y_lower_margin - y_upper_margin)/2))
						else:
							height_shift = 5 * distance * np.exp(0.001 * ((y_lower_margin - y_upper_margin)/2 - y_middle))
						width_shift = -angle_rotation * 0.35 / distance + 6
					elif magnitude == 2:
						if y_middle > (y_lower_margin - y_upper_margin)/2:
							height_shift = 12 * distance * np.exp(0.0015 * (y_middle - (y_lower_margin - y_upper_margin)/2))
						else:
							height_shift = 10 * distance * np.exp(0.001 * ((y_lower_margin - y_upper_margin)/2 - y_middle))
						width_shift = -angle_rotation * 0.5 / distance + 6
					else:
						if y_middle > (y_lower_margin - y_upper_margin) / 2:
							height_shift = 10 * distance * np.exp(0.0015 * (y_middle - (y_lower_margin - y_upper_margin) / 2)) - 16 * distance
						else:
							height_shift = 5 * distance * np.exp(0.001 * ((y_lower_margin - y_upper_margin) / 2 - y_middle)) - 16 * distance
						width_shift = -angle_rotation * 0.75 / distance + 6
					height_shift -= np.clip(np.abs(60 - angle_rotation), 0, 60) * magnitude * 0.125
			else:
				if wheel_class == 1:
					if magnitude == 1:
						height_shift = int(22 / distance) + int(-angle_rotation * 0.45) - 0.15 * (60 + angle_rotation) * np.tanh((350 - y_middle) * 0.01)
						width_shift = -angle_rotation * 0.5 / distance + 8
					elif magnitude == 2:
						height_shift = int(9 / distance) + int(-angle_rotation * 0.2) - 0.2 * (60 + angle_rotation) * np.tanh((500 - y_middle) * 0.01)
						width_shift = - angle_rotation * 0.55 / distance + 8
					else:
						height_shift = -int(3 / distance) + int(-angle_rotation * 0.1) - 0.33 * (60 + angle_rotation) * np.tanh((350 - y_middle) * 0.01)
						width_shift = -angle_rotation * 0.8 / distance + 8
					height_shift -= np.clip(np.abs(45 - angle_rotation), 0, 45) * magnitude * 0.1
				else:
					if magnitude == 1:
						if y_middle > (y_lower_margin - y_upper_margin) / 2:
							height_shift = 19 * distance * np.exp(0.0015 * (y_middle - (y_lower_margin - y_upper_margin) / 2))
						else:
							height_shift = 5 * distance * np.exp(0.001 * ((y_lower_margin - y_upper_margin) / 2 - y_middle))
						width_shift = - angle_rotation * 0.1 / distance + 10
					elif magnitude == 2:
						if y_middle > (y_lower_margin - y_upper_margin) / 2:
							height_shift = 12 * distance * np.exp(0.0015 * (y_middle - (y_lower_margin - y_upper_margin) / 2))
						else:
							height_shift = 10 * distance * np.exp(0.001 * ((y_lower_margin - y_upper_margin) / 2 - y_middle))
						width_shift = - angle_rotation * 0.5 / distance + 10
					else:
						if y_middle > (y_lower_margin - y_upper_margin) / 2:
							height_shift = 10 * distance * np.exp(0.0015 * (y_middle - (y_lower_margin - y_upper_margin) / 2)) - 12 * distance
						else:
							height_shift = 5 * distance * np.exp(0.001 * ((y_lower_margin - y_upper_margin) / 2 - y_middle)) - 22 * distance
						width_shift = - angle_rotation * 0.75 / distance + 10
					height_shift -= np.clip(np.abs(60 - angle_rotation), 0, 60) * magnitude * 0.125
			if check_collision(data_dict, int((x_middle - angle_rotation / 2) * (IMAGE_SIZE_WIDTH/background_width)),
			                   int((IMAGE_SIZE_WIDTH/background_width) * (y_middle - height_shift)), bbox_size - 30, bbox_size - 30):

				objects_list.append(rotated_wheel)
				delta_list.append((delta_h, delta_w))
				x_y_middle_list.append((x_middle, y_middle))

				data_dict[str(i)] = [int((x_middle + width_shift) * (IMAGE_SIZE_WIDTH/background_width)), int((IMAGE_SIZE_WIDTH/background_width) * int(y_middle + height_shift)),
				                     int(bbox_size * 0.9),
				                     wheel_class, magnitude,
				                     angle_rotation]
				break
	image_base = insert_objects(image_base, camera_central_point, objects_list, delta_list, x_y_middle_list)
	image_base = remove_white(image_base)

	object_mask = get_all_objects_mask(image_base)
	image_margins = add_margin_image(image_base, border_width, margin_thickness)
	image_base = add_object_shadows(image_base, data_dict)

	image_base += image_margins * (1 - np.transpose(np.repeat(object_mask, [3], axis=0), axes=[1, 2, 0]))
	aruco_tags = add_aruco_tags(image_base, border_width, margin_thickness, aruco_tags)
	image_base += aruco_tags * (1 - np.transpose(np.repeat(object_mask, [3], axis=0), axes=[1, 2, 0]))
	image_base = add_background_shadows(image_base)
	if random.random() >= 0.25:
		image_base = add_gripper(image_base, gripper)
	image_base = reshape(image_base, w=IMAGE_SIZE_WIDTH, h=IMAGE_SIZE_HEIGHT)

	image_base = image_base.astype(np.float32)
	#image_base = cv2.cvtColor(image_base, cv2.COLOR_BGR2RGB)
	#cv2.imwrite("test_image.png", image_base)

	#for key, value in data_dict.items():
		#image_base = plot_bbox(image_base, value[0], value[1], value[2], value[2], value[3])
	#plt.imshow(image_base/255)
	#plt.show()
	#exit()

	return image_base, data_dict


def generate_dataset():
	disk_magnitude_1 = cv2.imread(ROOT_PATH + "images/disk_magnitude_1.jpg")
	disk_magnitude_2 = cv2.imread(ROOT_PATH + "images/disk_magnitude_2.jpg")
	disk_magnitude_3 = cv2.imread(ROOT_PATH + "images/disk_magnitude_3.jpg")
	disk_dark_magnitude_1 = cv2.imread(ROOT_PATH + "images/disk_dark_magnitude_1.jpg")
	disk_dark_magnitude_2 = cv2.imread(ROOT_PATH + "images/disk_dark_magnitude_2.jpg")
	disk_dark_magnitude_3 = cv2.imread(ROOT_PATH + "images/disk_dark_magnitude_3.jpg")
	tyre_magnitude_1 = cv2.imread(ROOT_PATH + "images/tyre_magnitude_1.jpg")
	tyre_magnitude_2 = cv2.imread(ROOT_PATH + "images/tyre_magnitude_2.jpg")
	tyre_magnitude_3 = cv2.imread(ROOT_PATH + "images/tyre_magnitude_3.jpg")
	wheel_magnitude_1 = cv2.imread(ROOT_PATH + "images/wheel_magnitude_1.jpg")
	wheel_magnitude_2 = cv2.imread(ROOT_PATH + "images/wheel_magnitude_2.jpg")
	wheel_magnitude_3 = cv2.imread(ROOT_PATH + "images/wheel_magnitude_3.jpg")
	wheel_dark_magnitude_1 = cv2.imread(ROOT_PATH + "images/wheel_dark_magnitude_1.jpg")
	wheel_dark_magnitude_2 = cv2.imread(ROOT_PATH + "images/wheel_dark_magnitude_2.jpg")
	wheel_dark_magnitude_3 = cv2.imread(ROOT_PATH + "images/wheel_dark_magnitude_3.jpg")
	aruco_tags = list()
	for aruco_tag_index in range(10):
		aruco_tag = cv2.imread(ROOT_PATH + "/aruco_tags/aruco_tag{}.png".format(aruco_tag_index + 1))
		aruco_tags.append(aruco_tag)
	gripper = cv2.imread(ROOT_PATH + "images/gripper.png")
	count = 0
	for subset in ["training", "testing"]:
		file = open("detection_subset_{}.csv".format(subset), "w")
		csvwriter = csv.writer(file)
		all_rows = [["Image Path", "Labels Path"]]
		os.mkdir(ROOT_PATH + "wheels_detection_dataset_{}".format(subset))
		os.mkdir(ROOT_PATH + "wheels_detection_dataset_{}/images".format(subset))
		os.mkdir(ROOT_PATH + "wheels_detection_dataset_{}/labels".format(subset))

		for num in range(DATASET_SPLIT[count]):
			print("Image number {}".format(num))
			image, data_dictionary = generate_image([disk_magnitude_1,
			                                         disk_magnitude_2,
			                                         disk_magnitude_3,
			                                         disk_dark_magnitude_1,
			                                         disk_dark_magnitude_2,
			                                         disk_dark_magnitude_3,
			                                         tyre_magnitude_1,
			                                         tyre_magnitude_2,
			                                         tyre_magnitude_3,
			                                         wheel_magnitude_1,
			                                         wheel_magnitude_2,
			                                         wheel_magnitude_3,
			                                         wheel_dark_magnitude_1,
			                                         wheel_dark_magnitude_2,
			                                         wheel_dark_magnitude_3], aruco_tags, gripper)

			path_images = ROOT_PATH + "wheels_detection_dataset_{}/images/wheel_{}.png".format(subset, num)
			path_labels = ROOT_PATH + "wheels_detection_dataset_{}/labels/wheel_{}.txt".format(subset, num)
			cv2.imwrite(path_images, image)
			all_rows.append([path_images, path_labels])
			txt_file = open(path_labels, "w")

			for key in data_dictionary.keys():
				coord_x = data_dictionary[key][0] / IMAGE_SIZE_WIDTH
				coord_y = data_dictionary[key][1] / IMAGE_SIZE_HEIGHT
				bbox_size_x, bbox_size_y = data_dictionary[key][2] / IMAGE_SIZE_WIDTH, data_dictionary[key][2] / IMAGE_SIZE_HEIGHT
				label = data_dictionary[key][3]
				line = str(coord_x) + " " + str(coord_y) + " " + str(bbox_size_x) + " " + str(bbox_size_y) + " " + str(label) + "\n"
				txt_file.write(line)
		count += 1
		csvwriter.writerows(all_rows)
		file.close()


if __name__ == "__main__":
	generate_dataset()
