import cv2
import numpy as np
import time
import pydobot
import param

from block_detector import BlockDetector

# ============================================================
# CONFIG
# ============================================================
HOME_POSE = (230, 0, 170, 0)
APPROACH_Z = 120   # safe height
PICK_Z = -45       # adjust based on your setup

CAM_ID = param.camera_id()
H_FILE = "homography_matrix.npy"
CALIB_FILE = "calibration.npz"

# ============================================================
# INITIALIZE
# ============================================================
device = pydobot.Dobot(port=param.comport())
cam = cv2.VideoCapture(CAM_ID)

H = np.load(H_FILE)
detector = BlockDetector(CALIB_FILE)

print("Loaded homography:\n", H)

clicked_pixel = None


# ============================================================
# PIXEL → ROBOT TRANSFORM
# ============================================================
def pixel_to_robot(u, v):
    """Apply homography to pixel point (u, v) → robot (x, y)."""
    pt = np.array([u, v, 1.0])
    world = H @ pt
    world /= world[2]
    return float(world[0]), float(world[1])

# ============================================================
# CLICK CALLBACK
# ============================================================
def click_event(event, x, y, flags, param):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)
        print(f"\n🖱 Clicked pixel = ({x}, {y})")


cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", click_event)


# ============================================================
# MAIN LOOP
# ============================================================
print("\n>>> Click a block in the camera feed to move the robot there.\n")

while True:
    ok, frame = cam.read()
    if not ok:
        continue

    # For visualization
    disp = frame.copy()

    if clicked_pixel is not None:
        u, v = clicked_pixel

        # Draw clicked point
        cv2.circle(disp, (u, v), 5, (0, 255, 0), -1)

        # Convert to robot coordinates
        rx, ry = pixel_to_robot(u, v)
        cv2.putText(disp, f"({rx:.1f}, {ry:.1f})",
                    (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        print(f"→ Homography output: Robot XY = ({rx:.2f}, {ry:.2f})")

        # ======================================================
        # MOVE THE ROBOT THERE
        # ======================================================
        print("Moving robot...")

        rx+=0
        ry-=0

        # Approach position
        device.move_to(rx, ry, APPROACH_Z, 0)
        time.sleep(0.5)

        # Go down to pick
        device.move_to(rx, ry, PICK_Z, 0)
        time.sleep(0.5)

        # Suction grab
        device.suck(True)
        time.sleep(0.5)

        # Lift
        device.move_to(rx, ry, APPROACH_Z, 0)
        time.sleep(0.5)

        # Go home
        device.move_to(*HOME_POSE)
        time.sleep(0.5)

        print("✔ Complete!\n")

        clicked_pixel = None  # reset

    cv2.imshow("Camera", disp)

    if cv2.waitKey(1) & 0xFF == 27:
        break  # ESC quits


cam.release()
cv2.destroyAllWindows()
device.close()
