"""Render a labeled overview of the real MuJoCo terrain geometry."""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hybrid.arena import build_model
from .curriculum import curriculum
from .record import ROOT

CAPTIONS = (
    'Flat control: repeated reference surface',
    'Thresholds: 0.8 to 2.0 cm',
    'Up/down ramps: 2.5 to 6.5 cm rise',
    'Low steps: 2.4 to 6.0 cm peak',
    'Uneven blocks: up to 3.5 cm',
    'Cross-slope: 5.5 to 14.5 cm rise / 1.7 m',
    'Friction patch: coefficient 0.60 to 0.35',
    'Passage: inward sides, width decreases',
    'Moving crossing: 0.12 to 0.18 m/s',
    'Mixed: ramps, threshold, moving obstacle',
)


def render(output, variant=5):
    if variant not in range(6):
        raise ValueError('Variant must be in [0,5]')
    output = Path(output)
    if not output.is_absolute():
        output = ROOT/output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    width, height, caption = 480, 300, 55
    canvas = Image.new('RGB', (width*2, (height+caption)*5), '#101b26')
    font = ImageFont.load_default(size=16)
    for case in (c for c in curriculum() if c.variant == variant):
        model = build_model(case.scenario)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
        for i, obstacle in enumerate(case.scenario.obstacles):
            mid = int(model.body(f'obstacle_{i}').mocapid[0])
            data.mocap_pos[mid, :2] = obstacle.at(2.)
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.85, 0., .1]
        camera.distance, camera.azimuth, camera.elevation = 3.6, 125, -40
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(np.array(renderer.render(), copy=True))
        x, y = case.stage % 2*width, case.stage//2*(height+caption)
        canvas.paste(frame, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((x+10, y+height+4), f'Level {case.stage}: {case.family.replace("_", " ")}', font=font, fill='#79d6ed')
        draw.text((x+10, y+height+28), CAPTIONS[case.stage], font=font, fill='white')
    canvas.save(output)
    print(str(output), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/spike_pattern_skills/terrain-overview.png'))
    parser.add_argument('--variant', type=int, default=5)
    render(**vars(parser.parse_args()))
