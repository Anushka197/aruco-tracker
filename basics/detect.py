import cv2
import cv2.aruco as aruco
import json

# === Load config ===
with open("../config.json", "r") as f:
    config = json.load(f)

dict_name = config["aruco"]["dictionary"]
win_title = config["display"]["opencv_window"]["title"]
exit_key = config["display"]["opencv_window"]["exit_key"]
width = config["display"]["opencv_window"]["width"]
height = config["display"]["opencv_window"]["height"]
# === Init Dictionary & Detector ===
aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
parameters = aruco.DetectorParameters()

detector = aruco.ArucoDetector(aruco_dict, parameters)

# === Open webcam ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Detect markers ===
    corners, ids, _ = detector.detectMarkers(frame)

    # === Draw markers ===
    if ids is not None:
        for corner, marker_id in zip(corners, ids):
            pts = corner[0].astype(int)  # corner[0] is 4x2 array
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
            cv2.putText(frame, str(marker_id[0]), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


    # === Show the result ===
    cv2.imshow(win_title, cv2.resize(frame, (width, height)))

    # === Exit key ===
    if cv2.waitKey(1) & 0xFF == ord(exit_key):
        break

cap.release()
cv2.destroyAllWindows()
