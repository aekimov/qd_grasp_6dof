"""
Shadow Hand adduction/abduction + flexion demo
----------------------------------------------
Controls
  • q - quit
  • r - reset all joints
  • d - 10-s demo sequence

Update `shadow_hand_path` to point to your shadow_hand.urdf.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
from pathlib import Path

# ─────────────────────── CONFIG ────────────────────────
shadow_hand_path = "environments/3d_models/robots/shadow_hand.urdf"  # ← change me
TIME_STEP       = 1.0 / 240.0
DEMO_DURATION   = 10.0
# ────────────────────────────────────────────────────────

# Fresh GUI every run
if p.isConnected():
    p.disconnect()

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(TIME_STEP)
p.resetDebugVisualizerCamera(cameraDistance=0.5,
                             cameraYaw=45,
                             cameraPitch=-30,
                             cameraTargetPosition=[0, 0, 0])

p.loadURDF("plane.urdf")

if not Path(shadow_hand_path).is_file():
    raise FileNotFoundError(f"URDF not found: {shadow_hand_path}")
hand_id = p.loadURDF(shadow_hand_path, [0, 0, 0.5], useFixedBase=True)

# ─────────────────── Discover joints ────────────────────
adduct = {}   # J4 (abduction/adduction) joints
for j in range(p.getNumJoints(hand_id)):
    info = p.getJointInfo(hand_id, j)
    name = info[1].decode()
    if "J4" in name and info[2] == p.JOINT_REVOLUTE:
        adduct[name] = {"id": j, "lo": info[8], "hi": info[9]}
        print(f"  Ad/Ab joint {name:6s} id={j:2d} limits [{info[8]:+.3f}, {info[9]:+.3f}]")

flex = {        # extra flexion sliders we want
    "FFJ3": 10, "MFJ3": 15, "RFJ3": 20, "LFJ3": 26,
}

# ──────────────────── Create sliders ────────────────────
sliders = {}

def add_slider(label, lo, hi, jid):
    sid = p.addUserDebugParameter(label, min(lo, hi), max(lo, hi), 0.0)
    if sid < 0:
        print(f"[WARN] could not create slider '{label}'")
        return
    sliders[label] = (sid, jid)
    print(f"Created slider '{label}' id={sid}")

for n, d in adduct.items():
    add_slider(n, d["lo"], d["hi"], d["id"])

for n, jid in flex.items():
    lo, hi = (p.getJointInfo(hand_id, jid)[k] for k in (8, 9))
    add_slider(f"{n} (flexion)", lo, hi, jid)

print("\nSliders ready — press q/r/d as described above.\n")

# ────────────────────── Main loop ───────────────────────
demo_on, demo_t = False, 0.0

try:
    while p.isConnected():
        keys = p.getKeyboardEvents()

        if ord("q") in keys and keys[ord("q")] & p.KEY_IS_DOWN:
            break

        if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
            for lbl, (sid, jid) in sliders.items():
                p.resetJointState(hand_id, jid, 0)
                if sid >= 0:
                    p.resetDebugParameterValue(sid, 0)
            demo_on, demo_t = False, 0.0
            print("[INFO] reset")

        if ord("d") in keys and keys[ord("d")] & p.KEY_WAS_TRIGGERED:
            demo_on, demo_t = True, 0.0
            print("[INFO] demo started")

        if demo_on:
            demo_t += TIME_STEP
            wave = 0.3 * np.sin(2 * np.pi * 0.5 * demo_t)
            phase = {"FF": 0.0, "MF": np.pi/4, "RF": np.pi/2, "LF": 3*np.pi/4}

            for n, d in adduct.items():
                ph = next((p_ for k, p_ in phase.items() if k in n), 0.0)
                tgt = wave * np.sin(demo_t + ph)
                p.setJointMotorControl2(hand_id, d["id"],
                                        p.POSITION_CONTROL,
                                        targetPosition=tgt, force=5)

            if demo_t >= DEMO_DURATION:
                demo_on = False
                print("[INFO] demo finished")

        else:  # manual mode
            for lbl, (sid, jid) in list(sliders.items()):
                try:
                    if sid < 0:          # skip any failed slider
                        continue
                    tgt = p.readUserDebugParameter(sid)
                except p.error:
                    print(f"[WARN] cannot read slider '{lbl}' (id={sid})")
                    continue

                p.setJointMotorControl2(hand_id, jid,
                                        p.POSITION_CONTROL,
                                        targetPosition=tgt, force=5)

        p.stepSimulation()
        time.sleep(TIME_STEP)

except KeyboardInterrupt:
    pass

p.disconnect()
print("Bye!")
