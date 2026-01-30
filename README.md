# AI-Driven Robotic Manipulation: Vision, Planning, and Control with Dobot Magician Lite

This repo contains a small pipeline for:
1) **Camera calibration** (lens distortion),
2) **Block detection** (HSV color segmentation + square filtering),
3) **Pixel → robot coordinate mapping** via **homography**, and
4) A **main control loop** that detects blocks, asks an LLM for a plan, and executes moves on a **Dobot Magician Lite** using suction.

🎥 Demo video: https://www.youtube.com/watch?v=eA3R6aJcuzs&pp=2AYc

---

## Files

- **`main.py`** — Main interactive loop:
  - Reads camera frames on a background thread.
  - Detects colored blocks.
  - Converts detected pixel centers to robot XY using a saved homography matrix.
  - Sends `user_command`, `world_state`, and `detected_blocks` to an LLM for a JSON plan.
  - Executes the plan on the Dobot (move + suction).
- **`param.py`** — Your local configuration:
  - COM port for the Dobot
  - Camera device ID
- **`calib.py`** — Camera calibration:
  - Captures a checkerboard image, estimates camera matrix + distortion coefficients.
  - Saves to `calibration.npz`.
- **`block_detector.py`** — Block detector:
  - Undistorts frames using `calibration.npz`
  - HSV thresholding per color + contour filtering to find square blocks
  - Returns block centroids and approximate orientation.
- **`find_homography.py`** — Homography calibration:
  - Moves the robot to known world points, places a block, then detects that block in the camera frame.
  - Builds pixel↔world correspondences and computes/saves `homography_matrix.npy`.
- **`test_homography.py`** — Quick test tool:
  - Click in the camera window → shows homography output and moves robot to that XY (then picks with suction).

---

## Requirements

### Hardware
- Dobot Magician Lite (or compatible with `pydobot`)
- USB camera positioned to view the workspace
- Colored square blocks (yellow/red/blue/green work out of the box)

### Software
- Python 3.9+ recommended
- OpenCV
- NumPy
- `pydobot`
- `openai` Python SDK (only required for `main.py`)

Example install:
```bash
pip install numpy opencv-python pydobot openai
```

---

## 1) Configure ports & camera

Edit **`param.py`**:
```py
def comport():
    return "COM9"

def camera_id():
    return 1
```

- **Windows**: COM ports look like `COM3`, `COM9`, etc.
- **Linux/macOS**: you may need something like `/dev/ttyUSB0` (depending on `pydobot` / your driver).
- Camera ID is usually `0` or `1` depending on your machine.

---

## 2) Camera calibration (lens distortion)

Run:
```bash
python calib.py
```

Instructions:
- Hold a checkerboard in view.
- Press **SPACE** to capture.
- The script will save:
  - `calibration.npz` (camera matrix + distortion coefficients)

> If checkerboard detection fails, adjust `CHECKERBOARD` and/or ensure good lighting and focus.

---

## 3) Homography calibration (pixel → robot XY)

Run:
```bash
python find_homography.py
```

What it does:
- Uses a set of known robot **WORLD_POINTS** (mm).
- For each point:
  - Prompts you to feed a block
  - Picks it from HOME, places it at the world point
  - Takes a camera image and detects the block centroid
- Saves:
  - `homography_points.npy` (pixel/world pairs)
  - `homography_matrix.npy` (3×3 matrix used by the other scripts)

**Important:** The world points must match your actual physical setup and coordinate frame.
Adjust in `find_homography.py`:
```py
WORLD_POINTS = [
    (300, 100),
    (300, -100),
    (250, 100),
    (250, -100)
]
```

---

## 4) Test homography quickly

Run:
```bash
python test_homography.py
```

- A window opens showing the camera feed.
- Click a block; the script prints the mapped robot XY and moves the arm there, then attempts a suction pick.

Press **ESC** to quit.

> This is a fast sanity check before trying `main.py`.

---

## 5) Run the main GPT-controlled loop

Run:
```bash
python main.py
```

### Set your API key
`main.py` currently contains a placeholder:
```py
client = OpenAI(api_key="your-api-key-here")
```

Replace with your own key or set it via environment variables and load it safely.

### How it works
- You’ll be prompted:
  - `Enter command (or 'quit'):`
- For each command:
  1. Robot goes HOME, grabs the latest camera frame
  2. Detects blocks + converts pixel centroids to robot XY
  3. Sends info to the LLM
  4. Receives a JSON plan with:
     - `steps`: list of moves & suction toggles
     - `updated_world_state`: block IDs, positions, stacking relationships, history
  5. Executes the plan and updates state

### Safety / limits
`main.py` includes workspace constraints (x/y/z/angle ranges) in the prompt to the LLM.
Still:
- **Keep a hand near emergency stop / power**.
- Start with **slow speeds** and **large approach heights**.
- Ensure the camera view matches the workspace mapping used for the homography.

---

## Color detection notes

Default HSV ranges live in `block_detector.py`:
- yellow, red (two ranges), blue, green

If your lighting differs, you’ll likely need to tweak these ranges.

Also note:
- The detector expects blocks to be roughly square.
- Very small detections are ignored (`MIN_AREA`).

---

## Outputs & generated artifacts

After running the calibration steps, you should have:
- `calibration.npz` (from `calib.py`)
- `homography_matrix.npy` and `homography_points.npy` (from `find_homography.py`)

`main.py` expects these files to exist in the working directory.

---

## Troubleshooting

- **No camera image / black screen**
  - Try changing `camera_id()` in `param.py` to 0 or 1.
  - Ensure no other app is using the camera.

- **Dobot won’t connect**
  - Confirm the COM port and that drivers are installed.
  - Close other serial tools that might be holding the port.

- **Blocks not detected**
  - Improve lighting / reduce reflections.
  - Adjust HSV thresholds in `block_detector.py`.
  - Ensure blocks are within camera focus and not motion-blurred.

- **Robot misses block position**
  - Re-run `find_homography.py` after making sure the camera is fixed in place.
  - Check that your WORLD_POINTS match the physical coordinate system.
  - Make sure the table plane and camera view haven’t moved since calibration.

## Performance Notes

- **Detection Speed:** ~30 FPS on modern hardware
- **Planning Latency:** ~1.5-5 minutes (DEPENDENT ON THE LLM AND THE SCENE COMPLEXITY)
- **Positioning Accuracy:** ±2mm with proper calibration
- **Block Recognition:** >95% accuracy in good lighting
