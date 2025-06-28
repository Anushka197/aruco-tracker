import cv2
import pyvista as pv
import numpy as np
import threading
import json

# === Load config from JSON ===
with open('../config.json', 'r') as f:
    config = json.load(f)

# Extract config values
aruco_dict_name = config["aruco"]["dictionary"]
marker_length_mm = config["camera_params"]["mark_len"] / 1000  # Convert to meters

camera_matrix_file = config["camera_params"]["cam_mat"]
dist_coeffs_file = config["camera_params"]["dist_coeffs"]

cv_window_title = config["opencv_window"]["title"]
exit_key = config["opencv_window"]["exit_key"]

# === Load camera calibration ===
def load_camera_calibration(camera_matrix_file, dist_coeffs_file):
    camera_matrix = np.load(camera_matrix_file, allow_pickle=True)
    dist_coeffs = np.load(dist_coeffs_file, allow_pickle=True)
    return camera_matrix, dist_coeffs

camera_matrix, dist_coeffs = load_camera_calibration(camera_matrix_file, dist_coeffs_file)

# === PyVista Pyramid Model ===
def create_pyramid():
    points = np.array([
        [0.5, 0.5, 0],
        [-0.5, 0.5, 0],
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0, 0, 1]
    ])
    faces = np.hstack([[4, 0, 1, 2, 3],
                       [3, 0, 1, 4],
                       [3, 1, 2, 4],
                       [3, 2, 3, 4],
                       [3, 3, 0, 4]])
    return pv.PolyData(points, faces)

pyramid_model = create_pyramid()
original_pyramid_model = pyramid_model.copy()

plotter = pv.Plotter()
pyramid_actor = plotter.add_mesh(pyramid_model)
plotter.show_axes()
plotter.show(interactive_update=True, auto_close=False)

# === ArUco Detection ===
aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
parameters = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

running = True
frame = None

# === Capture thread ===
def capture_frames():
    global running, frame
    while running:
        ret, temp_frame = cap.read()
        if not ret:
            running = False
            break
        frame = temp_frame

capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

frame_counter = 0

# === Main loop ===
while running:
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for corner in corners:
            pyramid_model.copy_from(original_pyramid_model)
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner, marker_length_mm, camera_matrix, dist_coeffs
            )
            rotation_matrix, _ = cv2.Rodrigues(rvec[0])
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = tvec[0].flatten()
            pyramid_model.transform(transformation_matrix)

    plotter.update()
    cv2.imshow(cv_window_title, frame)
    if cv2.waitKey(1) & 0xFF == ord(exit_key):
        running = False

    frame_counter += 1

# === Cleanup ===
capture_thread.join()
cap.release()
cv2.destroyAllWindows()
plotter.close()
