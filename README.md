# Delta-Robot-AI
Wheel detection based on CNNs and fiducial markers for plane detection. The frames are streamed from the delta controller using 5G.

![alt text](./assets/plane_detection_demo.gif)

## Features

1. **Plane Detection**
   - Uses ArUco/AprilTag markers for plane recognition.
   - Calculates homography transformations for planar mapping.
   - Added under the `plane_detector` [submodule](https://github.com/david-s-martinez/Automatic-Workspace-Calibration-Based-on-Aruco).

2. **Object Detection**  
   - Employs a neural network (e.g., MobileNet or ResNet) for object detection.
   - Supports object tracking and visualization with bounding boxes and confidence scores.

3. **Data Communication**  
   - Posts detection results to a server endpoint in JSON format.

4. **Real-Time Performance**  
   - Processes frames from a camera stream and displays results in real time.
   - Uses multi-threading and multiprocessing to optimize performance.

---

## Requirements

### Libraries
- Python 3.9
- [PyTorch](https://pytorch.org/)
- OpenCV (`cv2`)
- NumPy
- Requests
- Custom modules:
  - `Detection_models`
  - `PlaneDetection` from `plane_computation.plane_detection`
  - `test_model_Delta_v3` from `conv_net_detect`

### Hardware
- A camera device (Raspberry Pi Camera or other video sources).
- GPU for running neural networks (CUDA-supported).

---

## Setup

1. **Clone the repository with submodule**  
   ```bash
   git clone --recursive  <repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies**  
   Ensure Python dependencies are installed:
   ```bash
   pip install <missing module>
   ```

3. **Prepare Configuration Files**  
   Add the following required configuration files to the specified paths:
   - **Camera Configuration:**
     - `camera_matrix_rpi.txt`
     - `distortion_rpi.txt`
   - **Plane Points:**  
     - `plane_points_new_tray.json`
     - `plane_points_old_tray.json`
   - **Neural Network Model:**  
     - Pre-trained weights (`MOBILENET_V2_FINER_GRID_2_weights_saved.pt` or `RESNET_18_FINER_GRID_2_weights_saved.pt`).

4. **Directory Structure**  
   Ensure the following structure:
   ```
   .
   ├── conv_net_detect/
   │   ├── disk_centroid_template_1.png
   │   ├── disk_centroid_template_2.png
   │   └── disk_centroid_template_3.png
   ├── plane_computation/
   │   └── plane_detection.py
   ├── model_configs/
   │   └── [MODEL_WEIGHTS_FILES]
   ├── vision_configs/
   │   └── [CAMERA_CONFIG_FILES]
   └── main.py
   ```

---

## Usage

1. **Run the Application**  
   ```bash
   python capture_stream.py
   ```

2. **Key Operations**  
   - `ESC`: Stop the program.
   - `S`: Save frames for debugging or analysis.

3. **Modify Parameters**  
   Update the parameters in the `config` dictionary to adapt to your setup:
   - Change `IS_ONLINE` to toggle between a live camera and a video file.
   - Update paths for models and camera configurations.

---

## Components Overview

### `robot_perception`
Processes camera frames, performs plane and object detection, and overlays the results.

### `cam_reader`
Captures video frames from the camera or video source.

### `post_detections`
Posts detection data to a specified server endpoint.

### `plot_boxes`
Draws bounding boxes and labels on detected objects.

---

## Troubleshooting

- **Camera Not Opening:**  
  Check `cam_source` in the configuration.
  
- **Model Loading Issues:**  
  Ensure the pre-trained model weights are in the correct path.

- **Slow Performance:**  
  Use a GPU and verify CUDA installation.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- Darknet2PyTorch library for object detection.
- OpenCV for computer vision operations.
- PyTorch for deep learning models.