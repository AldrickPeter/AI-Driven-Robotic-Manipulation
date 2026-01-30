import json
import time
import numpy as np
import cv2
import threading

import pydobot
import param
from openai import OpenAI
from block_detector import BlockDetector

# ============================================================
# GLOBAL SHARED FRAME
# ============================================================
latest_frame = None
frame_lock = threading.Lock()
running = True

# ============================================================
# CAMERA THREAD
# ============================================================
def camera_thread_func(cam_id):
    global latest_frame, running

    cam = cv2.VideoCapture(cam_id)

    while running:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.01)
            continue

        with frame_lock:
            latest_frame = frame.copy()

        # Optional: local view window
        _, annotated = detector.process_frame(frame, draw=True)
        cv2.imshow("Live Feed", annotated)
        if cv2.waitKey(1) == ord('q'):
            running = False
            break

    cam.release()
    cv2.destroyAllWindows()

# ============================================================
# CONFIG
# ============================================================
HOME = (230, 0, 170, 0)
CAM_ID = param.camera_id()

ROBOT_LIMITS = {
    "x_min": 150, "x_max": 350,
    "y_min": -150, "y_max": 150,
    "z_min": -60, "z_max": 170,
    "angle_min": -180, "angle_max": 180
}

BLOCK_HEIGHT = 10

# ============================================================
# INITIALIZATION
# ============================================================
H = np.load("homography_matrix.npy")
detector = BlockDetector("calibration.npz")
device = pydobot.Dobot(port=param.comport())

client = OpenAI(api_key="your-api-key-here")
device.move_to(*HOME)

# ============================================================
# THREAD START
# ============================================================
camera_thread = threading.Thread(target=camera_thread_func, args=(CAM_ID,), daemon=True)
camera_thread.start()

# ============================================================
# WORLD STATE
# ============================================================
world_state = {
    "next_block_id": 1,
    "blocks": {},
    "history": []
}

# ============================================================
# PIXEL → ROBOT XY
# ============================================================
def pixel_to_robot(u, v):
    pt = np.array([u, v, 1.0])
    world = H @ pt
    world /= world[2]
    return float(world[0]), float(world[1])

# ============================================================
# EXECUTE ROBOT STEPS
# ============================================================
def execute_step(step):
    print("\nExecuting step:", step)
    action = step["action"]

    if action == "move":
        device.move_to(step["x"], step["y"], step["z"], step["angle"])
        time.sleep(0.2)

    elif action == "suck":
        device.suck(step["state"])
        time.sleep(0.2)

# ============================================================
# MAIN LOOP
# ============================================================
while running:

    cmd = input("\nEnter command (or 'quit'): ")
    if cmd == "quit":
        running = False
        break

    # --------------------------------------------------------
    # 1. GET FRAME FROM CAMERA THREAD
    # --------------------------------------------------------
    device.move_to(*HOME)
    time.sleep(1.5)

    with frame_lock:
        frame_copy = None if latest_frame is None else latest_frame.copy()

    if frame_copy is None:
        print("Waiting for camera...")
        time.sleep(0.1)
        continue

    # --------------------------------------------------------
    # 2. DETECT BLOCKS
    # --------------------------------------------------------
    results, _ = detector.process_frame(frame_copy, draw=False)

    detected_blocks = []
    for b in results["blocks"]:
        rx, ry = pixel_to_robot(b["cx"], b["cy"])
        detected_blocks.append({
            "color": b["color"],
            "x": rx,
            "y": ry,
            "angle": b["angle_deg"]
        })

    # --------------------------------------------------------
    # 3. GPT PLANNING
    # --------------------------------------------------------
    system_prompt = f"""
You are the motion planner AND world-state manager for a Dobot Magician Lite robot.

==================== ROBOT COMMAND RULES ====================
Your output MUST be JSON with 2 keys:
1. "steps": list of robot commands
2. "updated_world_state": the new full world model

Allowed robot commands:
- {{ "action": "move", "x": <num>, "y": <num>, "z": <num>, "angle": <num> }}
- {{ "action": "suck", "state": true|false }}

Robot safety limits:
- x: {ROBOT_LIMITS['x_min']} to {ROBOT_LIMITS['x_max']}
- y: {ROBOT_LIMITS['y_min']} to {ROBOT_LIMITS['y_max']}
- z: {ROBOT_LIMITS['z_min']} to {ROBOT_LIMITS['z_max']}
- angle: 0 to 180

General Direction Description:
- The robot's right side is -Y, left side is +Y.
- The higher the X value, the farther from the robot base.

Always approach from above (>80mm), then descend.

==================== BLOCK / Z HEIGHT RULES ====================
- Try to pick up the block that is the closest to you (smallest x).
- When stacking blocks, compute the correct z height:
- Table height = z = -60.
- Picking an unstacked block happens at z = -50.
- Every stacked block adds +10.
- ALWAYS compute stacking heights exactly: base_z + 10.
- When placing a block:
    - approach at target_z + 80
    - descend to target_z
    - release at exactly target_z
    - ascend back up

==================== OBJECT IDENTITY MANAGEMENT ====================
You receive "detected_blocks" from the camera with raw positions.

You are fully responsible for assigning and maintaining block IDs.

World State Format:
- "next_block_id": integer
- "blocks": {{
      "block_001": {{ "color":..., "x":..., "y":..., "z":..., "angle":..., "on_top_of":... }}
  }}
- "history": list of events, each with a short description for your future reference.

Matching detections:
A detected block matches an existing block if:
    |x_detected - x_world| < 25 mm AND
    |y_detected - y_world| < 25 mm

If NO match:
    1. Assign new ID: block_XXX where XXX = next_block_id zero-padded.
    2. Increment next_block_id.
    3. Add a history entry:
        {{"event": "new_block", "block_id": "...", "color": "..." }}

If matched:
    - Update that block's x, y, angle.

Never rename or delete IDs.
Never create blocks that were not detected.
Never merge blocks.

==================== IMPORTANT ====================
Output ONLY valid JSON.
""" 

    user_prompt = json.dumps({
        "user_command": cmd,
        "world_state": world_state,
        "detected_blocks": detected_blocks
    })

    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        reasoning_effort="low"
    )

    plan = json.loads(resp.choices[0].message.content)
    print("\nGPT PLAN:\n", json.dumps(plan, indent=2))

    # --------------------------------------------------------
    # 4. EXECUTE ROBOT PLAN
    # --------------------------------------------------------
    for step in plan["steps"]:
        execute_step(step)
        time.sleep(0.5)

    # --------------------------------------------------------
    # 5. UPDATE WORLD STATE
    # --------------------------------------------------------
    world_state = plan["updated_world_state"]
    device.move_to(*HOME)

# END MAIN LOOP
running = False
camera_thread.join()
print("Shutting down...")