import cv2
import os

CHECKERBOARD = (9, 6)
cap = cv2.VideoCapture(0)
save_dir = "calib_images"
os.makedirs(save_dir, exist_ok=True)
count = 0

print("Press SPACE to save image, Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if found:
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, found)

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Spacebar to save
        cv2.imwrite(os.path.join(save_dir, f"img_{count}.png"), frame)
        print(f"Saved img_{count}.png")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
