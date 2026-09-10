# Go2 MuJoCo Simulation

A project for simulating the Unitree Go2 quadruped in MuJoCo.

## Setup

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Create and activate a Python virtual environment for your operating system.
4. Install all dependencies:

   python -m pip install -r requirements.txt

5. Launch the simulation:

   python -m mujoco.viewer --mjcf unitree_go2/scene.xml

On macOS, use `mjpython` instead of `python` for the viewer command.

## Model source

The Go2 model is from [Google DeepMind's MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2).
Its BSD-3-Clause license is included in `unitree_go2/LICENSE`.