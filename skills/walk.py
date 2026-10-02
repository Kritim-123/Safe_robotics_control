"""Reusable forward-walking skill for MuJoCo simulation."""
from pathlib import Path
import argparse
import math
from controllers.walking import WalkingExperiment

ROOT = Path(__file__).resolve().parents[1]


def walk(duration=10.0, viewer=True):
    """Walk forward in a fresh simulation; return (history, summary).

    Duration excludes two seconds of standing preparation. Each call resets
    the robot. This skill does not yet support chaining or hardware control.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    experiment = WalkingExperiment(
        ROOT / "unitree_go2" / "scene_indoor.xml",
        kp=90.0, ki=3.0, kd=3.0,
        period=0.6, stride=0.08, lift=0.06, stance=0.60,
    )
    return experiment.run(mode="walk", duration=duration, viewer=viewer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    _, summary = walk(duration=args.duration, viewer=not args.headless)
    print(f"Stopped: {summary['reason']}")
    print(f"Forward: {summary['forward']:.3f} m")
    print(f"Sideways: {summary['lateral']:.3f} m")
    print(f"Body height: {summary['height']:.3f} m")


if __name__ == "__main__":
    main()
