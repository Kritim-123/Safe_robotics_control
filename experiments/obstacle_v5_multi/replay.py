"""Replay assessment with a new evidence bank, preserving retained results."""
import argparse
from pathlib import Path

from . import evaluate
from experiments.obstacle_v5_refinement.model import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    if not folder.is_relative_to(ROOT/'output'):
        raise ValueError('Keep replay evidence inside output/')
    evaluate.ASSETS = folder.parent/(folder.name+'-artifacts')
    evaluate.ASSETS.mkdir(parents=True, exist_ok=False)
    evaluate.main()


if __name__ == '__main__':
    main()
