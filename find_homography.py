import cv2
import numpy as np
import time
from pydobot.dobot import MODE_PTP
import pydobot
import param

from block_detector import BlockDetector

# ============================================================
# CONFIG
# ============================================================
HOME = (230, 0, 170, 0)   # Camera snapshot pose
APPROACH_Z = 140
DROP_Z = -49

# Known world coordinates of your calibration squares
WORLD_POINTS = [
    (300, 100),
    (300, -100),
    (250, 100),
    (250, -100)
]

CALIB_FILE = "calibration.npz"
H_POINTS_FILE = "homography_points.npy"
H_FILE = "homography_matrix.npy"

NUM = 4  # number of points

# PARAMETERS FOR NEW-POINT VALIDATION
MIN_NEW_DIST_PX = 30    # minimum pixel distance from any previously accepted point
MAX_RETRIES = 4         # retries for capturing a valid new point
RETRY_DELAY = 0.8       # seconds between retries

# ============================================================
# INIT
# ============================================================
device = pydobot.Dobot(port=param.comport())
detector = BlockDetector(CALIB_FILE)
   # open camera once

pixel_pts = []
world_pts = []

def move(x, y, z, r=0, wait=0.5):
    device.move_to(x, y, z, r)
    time.sleep(wait)

def euclid(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def select_new_centroid(all_centroids, existing_pixels, min_dist_px):
    """
    Choose a centroid from all_centroids that is 'new' w.r.t existing_pixels.
    Strategy: compute for each candidate its minimum distance to existing_pixels,
    then choose the candidate with the largest such min-distance. Accept only if
    that distance >= min_dist_px.
    Returns (cx,cy) or None if none accepted.
    """
    if len(all_centroids) == 0:
        return None

    if len(existing_pixels) == 0:
        # no previous points, pick the largest (or first)
        return tuple(all_centroids[0])

    best = None
    best_min_dist = -1.0

    for c in all_centroids:
        # min distance from this candidate to any existing pixel
        dists = [euclid(c, p) for p in existing_pixels]
        min_d = min(dists) if len(dists) > 0 else float('inf')
        if min_d > best_min_dist:
            best_min_dist = min_d
            best = tuple(c)

    if best is not None and best_min_dist >= min_dist_px:
        return best
    else:
        return None

# ============================================================
# MAIN CALIBRATION LOOP
# ============================================================
print("\n============================")
print("  HOMOGRAPHY CALIBRATION")
print("============================\n")

for i in range(NUM):

    world_xy = WORLD_POINTS[i]
    print("\n========================================")
    print(f"          BLOCK {i+1} / {NUM}")
    print("========================================")
    print("Feed a block to the gripper, then press ENTER.")
    input()

    # 1. Go HOME and suck ON (ready to pick)
    move(*HOME)
    device.suck(True)
    time.sleep(0.4)

    # 2. Pick block at HOME exactly
    move(HOME[0], HOME[1], APPROACH_Z)
    move(HOME[0], HOME[1], DROP_Z)
    time.sleep(0.5)
    move(HOME[0], HOME[1], APPROACH_Z)

    # 3. Move to known world position and drop
    print(f"Placing block at WORLD point: {world_xy}")
    wx, wy = world_xy

    move(wx, wy, APPROACH_Z)
    move(wx, wy, DROP_Z)

    # Drop block
    device.suck(False)
    time.sleep(0.5)

    # Lift away from block
    move(wx, wy, APPROACH_Z)

    # 4. Return home to take a picture of the scene
    print("Returning HOME to capture image...")
    move(*HOME)

    time.sleep(2)

    # 5. Try to capture a new pixel (retry if duplicate / not found)
    new_centroid = None
    cam = cv2.VideoCapture(param.camera_id())
    for attempt in range(1, MAX_RETRIES + 1):
        ok, frame = cam.read()
        if not ok:
            print(f"❌ Camera read failed on attempt {attempt}.")
            time.sleep(RETRY_DELAY)
            continue

        results, annotated = detector.process_frame(frame, draw=True)

        # Extract centroids of all detections
        all_centroids = [(b["cx"], b["cy"]) for b in results["blocks"]]

        # Show annotated frame briefly so user can see what's detected
        cv2.imshow("Calibration Detection", annotated)
        cv2.waitKey(300)

        new_centroid = select_new_centroid(all_centroids, pixel_pts, MIN_NEW_DIST_PX)
        if new_centroid is not None:
            print(f"✔ Found new centroid at attempt {attempt}: {new_centroid}")
            break
        else:
            # Nothing qualifies as new: show info and retry
            if len(all_centroids) == 0:
                print(f"Attempt {attempt}: No blocks detected in frame.")
            else:
                print(f"Attempt {attempt}: Detected centroids {all_centroids}, but none are sufficiently different from previous points {pixel_pts}.")
            time.sleep(RETRY_DELAY)

    if new_centroid is None:
        # final fallback: if there are detections, pick the centroid farthest from existing points
        ok, frame = cam.read()
        if ok:
            results, annotated = detector.process_frame(frame, draw=True)
            all_centroids = [(b["cx"], b["cy"]) for b in results["blocks"]]
            if all_centroids:
                # pick the centroid with maximum min-distance (even if < threshold)
                fallback = select_new_centroid(all_centroids, pixel_pts, -1)
                if fallback is not None:
                    new_centroid = fallback
                    print("⚠ Fallback: accepting best candidate even though it's close to previous points:", new_centroid)
        if new_centroid is None:
            print("❌ Could not find a valid new centroid for this placement. Skipping this point.")
            # Optionally: you could choose to continue (skip) or abort; we skip record and continue.
            continue
    cam.release()

    u, v = new_centroid
    print(f"✔ Pixel = ({u}, {v})  ↔  World = {world_xy}")

    pixel_pts.append([int(u), int(v)])
    world_pts.append([wx, wy])

# ============================================================
# SAVE POINT CORRESPONDENCES
# ============================================================
np.save(H_POINTS_FILE, {"pixels": pixel_pts, "world": world_pts})
print("\nSaved:", H_POINTS_FILE)

# ============================================================
# COMPUTE HOMOGRAPHY
# ============================================================
if len(pixel_pts) < 4:
    print("\n❌ NOT ENOUGH POINTS FOR HOMOGRAPHY! (found {})".format(len(pixel_pts)))
else:
    pixels = np.array(pixel_pts, dtype=np.float32)
    world = np.array(world_pts, dtype=np.float32)

    print("\nComputing homography...")
    H, mask = cv2.findHomography(pixels, world, method=0)

    print("\n==============================")
    print("        HOMOGRAPHY MATRIX H")
    print("==============================\n")
    print(H)

    np.save(H_FILE, H)
    print("\nSaved H to:", H_FILE)

# Cleanup
cam.release()
cv2.destroyAllWindows()
device.close()
