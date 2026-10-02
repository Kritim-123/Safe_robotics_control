"""Move the simulated Go2 through explicit waypoints using classical control."""
import argparse
import json

from controllers.navigation import NavigationExperiment


def navigate(goal=(1.0, 0.0), *, waypoints=None, start=(0.0, 0.0),
             start_yaw=0.0, timeout=120.0, tolerance=0.12, viewer=True):
    """Run one episode. Coordinates are world-frame metres; yaw is radians.

    When provided, waypoints precede goal. Paths must be clear; no avoidance
    or planning is implemented. Returns (history, summary), including success.
    """
    route = [] if waypoints is None else list(waypoints)
    route.append(goal)
    return NavigationExperiment().navigate(route, start=start, start_yaw=start_yaw,
        timeout=timeout, tolerance=tolerance, viewer=viewer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal', nargs=2, type=float, default=(1.0, 0.0), metavar=('X', 'Y'))
    parser.add_argument('--start', nargs=2, type=float, default=(0.0, 0.0), metavar=('X', 'Y'))
    parser.add_argument('--start-yaw', type=float, default=0.0)
    parser.add_argument('--waypoint', nargs=2, type=float, action='append', default=[])
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    _, summary = navigate(args.goal, waypoints=args.waypoint, start=args.start,
                          start_yaw=args.start_yaw, timeout=args.timeout,
                          viewer=not args.headless)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary['success'] else 1)


if __name__ == '__main__':
    main()
