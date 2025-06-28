import cv2
import cv2.aruco as aruco
import json
import os

# === Load Config ===
with open("../config.json", "r") as f:
    config = json.load(f)

dict_name = config["aruco"]["dictionary"]
marker_size = config["aruco"]["size_px"]

# === Resolve dictionary from string ===
aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))

# === Output folder ===
output_dir = "D:\\AnushkaData\\DA life\\ArucoProjects\\varnavat\\images"
os.makedirs(output_dir, exist_ok=True)

# === Generate markers from ID 0 to 2 ===
for marker_id in range(3):
    marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    file_path = os.path.join(output_dir, f"aruco_id_{marker_id}.png")
    cv2.imwrite(file_path, marker_img)
    print(f"Saved: {file_path}")
