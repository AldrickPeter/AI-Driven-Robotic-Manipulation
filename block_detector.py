import cv2
import numpy as np
import math
import param

# ============================================================
# CONFIG (same values as before)
# ============================================================
COLOR_RANGES = {
    "yellow": ((20, 120, 120), (35, 255, 255)),
    "red1":   ((0, 120, 120), (10, 255, 255)),
    "red2":   ((170, 120, 120), (180, 255, 255)),
    "blue":   ((95, 120, 120), (130, 255, 255)),
    "green":  ((38, 100, 100), (90, 255, 255)),
}

DRAW_COLOR = {
    "yellow": (0, 255, 255),
    "red":    (0, 0, 255),
    "blue":   (255, 0, 0),
    "green":  (0, 255, 0),
}

MIN_AREA = 1000
SQUARE_ASPECT_TOL = 0.30
APPROX_EPS_FACTOR = 0.02


# ============================================================
# Utility Functions
# ============================================================
def load_calibration(path):
    d = np.load(path)
    return d["camera_matrix"], d["dist_coeffs"]


def postprocess_mask(mask):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def is_square_by_contour(cnt):
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return False

    rect = cv2.minAreaRect(cnt)
    (_, _), (w, h), _ = rect

    if w == 0 or h == 0:
        return False

    ratio = min(w, h) / max(w, h)
    if ratio < (1 - SQUARE_ASPECT_TOL):
        return False

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, APPROX_EPS_FACTOR * peri, True)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        return True

    return True


def normalize_angle_0_90(rect):
    (_, _), (w, h), angle = rect
    raw_angle = angle

    if w < h:
        raw_angle += 90

    norm_angle = abs(raw_angle) % 180
    if norm_angle > 90:
        norm_angle = 180 - norm_angle

    return norm_angle

def normalize_angle_0_45(rect):
    (_, _), (w, h), angle = rect

    # fix OpenCV minAreaRect conventions
    if w < h:
        angle += 90

    angle = abs(angle) % 180  # canonical 0–180

    # meaningful angle for a square
    return abs(angle % 45)


# ============================================================
# Detector Class
# ============================================================
class BlockDetector:
    def __init__(self, calib_file):
        self.camera_matrix, self.dist_coeffs = load_calibration(calib_file)

    def undistort(self, frame):
        h, w = frame.shape[:2]
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1
        )
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_mtx)

    def build_masks(self, hsv):
        masks = {}

        for name, rng in COLOR_RANGES.items():
            if name in ["red1", "red2"]:
                continue
            low, high = rng
            masks[name] = cv2.inRange(hsv, np.array(low), np.array(high))

        red_mask = cv2.inRange(hsv, np.array(COLOR_RANGES["red1"][0]),
                               np.array(COLOR_RANGES["red1"][1])) | \
                   cv2.inRange(hsv, np.array(COLOR_RANGES["red2"][0]),
                               np.array(COLOR_RANGES["red2"][1]))

        masks["red"] = red_mask

        return masks

    def process_frame(self, frame, draw=True):
        """
        Process one frame.
        Returns:
            {
                "blocks": [
                    {
                        "color": str,
                        "cx": int,
                        "cy": int,
                        "angle_deg": float,
                        "box": [[x,y], ...]
                    }
                ]
            },
            annotated_frame
        """
        undist = self.undistort(frame)
        hsv = cv2.cvtColor(undist, cv2.COLOR_BGR2HSV)
        output = undist.copy()

        masks = self.build_masks(hsv)

        results = {"blocks": []}

        for cname, mask in masks.items():
            mask = postprocess_mask(mask)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if not is_square_by_contour(c):
                    continue

                rect = cv2.minAreaRect(c)
                angle = normalize_angle_0_90(rect)

                # box points
                box = cv2.boxPoints(rect).astype(int).tolist()

                # centroid via moments
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    (cx, cy) = rect[0]

                # Append result (clean for LLM)
                results["blocks"].append({
                    "color": cname,
                    "cx": int(cx),
                    "cy": int(cy),
                    "angle_deg": float(angle),
                    "box": box,
                })

                # Draw if needed
                if draw:
                    cv2.drawContours(output, [np.array(box)], 0, DRAW_COLOR[cname], 2)
                    cv2.circle(output, (cx, cy), 3, (255, 255, 255), -1)
                    txt = f"{cname} {angle:.1f}°"
                    cv2.putText(output, txt, (cx + 8, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, DRAW_COLOR[cname], 2)

        return results, output

if __name__ == "__main__":
    DETECTOR = BlockDetector("calibration.npz")

    cam = cv2.VideoCapture(param.camera_id())

    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        results, annotated = DETECTOR.process_frame(frame, draw=True)

        cv2.imshow("Block Detection", annotated)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

    cam.release()
    cv2.destroyAllWindows()