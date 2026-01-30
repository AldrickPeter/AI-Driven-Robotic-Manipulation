import cv2
import numpy as np
import time
import pydobot
import param

# -----------------------------
# CONFIGURATION
# -----------------------------
CHECKERBOARD = (8, 6)       # inner corners (adjust to your pattern)
SQUARE_SIZE = 25          # arbitrary, set to 1.0 unless real scale is needed
CAMERA_ID = param.camera_id()         # default webcam

SAVE_FILE = "calibration.npz"

# device = pydobot.Dobot(port=param.comport())
# device.move_to(230, 0, 170, 0)  # Move to home position
# time.sleep(2)  # wait for the robot to reach the position  

# -----------------------------
# CAPTURE CHECKERBOARD IMAGE
# -----------------------------
cap = cv2.VideoCapture(CAMERA_ID)
print("Press SPACE to capture the checkerboard image...")
img = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Live Feed - Show Checkerboard", frame)
    key = cv2.waitKey(1)

    if key == 32:  # SPACE pressed
        img = frame.copy()
        print("Captured!")
        break
    elif key == 27:  # ESC to quit
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# FIND CHECKERBOARD CORNERS
# -----------------------------
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if not ret:
    print("Checkerboard not found! Try again.")
    exit()

# refine corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_refined = cv2.cornerSubPix(
    gray, corners, (11, 11), (-1, -1), criteria
)

# Draw corners
cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
cv2.imshow("Detected Corners", img)
cv2.waitKey(1000)

# -----------------------------
# PREPARE CALIBRATION DATA
# -----------------------------
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = [objp]
imgpoints = [corners_refined]

# -----------------------------
# CALIBRATE CAMERA
# -----------------------------
h, w = gray.shape
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (w, h), None, None
)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# -----------------------------
# SAVE CALIBRATION DATA
# -----------------------------
np.savez(SAVE_FILE, camera_matrix=mtx, dist_coeffs=dist)
print(f"Calibration saved to {SAVE_FILE}")

# -----------------------------
# UNDISTORT USING THE RESULT
# -----------------------------
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
