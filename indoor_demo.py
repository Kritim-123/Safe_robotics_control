"""Inspect the indoor scene with a stationary joint PD posture controller."""

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np


SCENE = Path(__file__).resolve().parent / "unitree_go2" / "scene_indoor.xml"


def load_scene():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    # Menagerie's home ctrl values are joint angles; these actuators take torque.
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    return model, data


def hold_posture(model, data, target):
    joints = model.actuator_trnid[:, 0]
    q = data.qpos[model.jnt_qposadr[joints]]
    dq = data.qvel[model.jnt_dofadr[joints]]
    data.ctrl[:] = np.clip(50 * (target - q) - 3 * dq,
                           model.actuator_ctrlrange[:, 0],
                           model.actuator_ctrlrange[:, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run a 10-second posture smoke check")
    args = parser.parse_args()
    model, data = load_scene()
    target = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]].copy()
    if args.headless:
        for _ in range(round(10 / model.opt.timestep)):
            hold_posture(model, data, target)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or data.qpos[2] < 0.18:
                raise RuntimeError("Posture check failed: robot fell or state became nonfinite")
        print(f"10 s posture check passed; base height: {data.qpos[2]:.3f} m")
        return

    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [2, 0, 0.3]
        viewer.cam.distance = 12
        viewer.cam.azimuth = -115
        viewer.cam.elevation = -65
        while viewer.is_running():
            start = time.perf_counter()
            with viewer.lock():
                hold_posture(model, data, target)
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
