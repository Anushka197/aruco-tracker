import cv2
import cv2.aruco as aruco
import numpy as np
import pickle
import json

# === Load config ===
with open("../config.json", "r") as f:
    config = json.load(f)

dict_name = config["aruco"]["dictionary"]
win_title = config["display"]["opencv_window"]["title"]
exit_key = config["display"]["opencv_window"]["exit_key"]
width = config["display"]["opencv_window"]["width"]
height = config["display"]["opencv_window"]["height"]
marker_length = config["transformation"]["pose_estimation"]["marker_length_meters"]

# Load camera parameters
with open(config["camera"]["calibration_files"]["camera_matrix"], "rb") as f:
    camera_matrix = pickle.load(f)

with open(config["camera"]["calibration_files"]["dist_coeffs"], "rb") as f:
    dist_coeffs = pickle.load(f)

# Init ArUco detector
aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)    

    cv2.imshow(win_title, frame)
    if cv2.waitKey(1) & 0xFF == ord(exit_key):
        break

cap.release()
cv2.destroyAllWindows()
