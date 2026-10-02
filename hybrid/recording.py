"""Optional offscreen MuJoCo recording; no effect on simulation control time."""
from pathlib import Path

import mujoco
import numpy as np


class VideoRecorder:
    def __init__(self, path, *, playback_speed=4, fps=20):
        import imageio.v2 as imageio
        if playback_speed <= 0 or fps <= 0:
            raise ValueError('Video speed and fps must be positive')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=fps, codec='libx264', quality=7,
                                        macro_block_size=16, ffmpeg_log_level='error')
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [1.4, 0, .15]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 6.4, 90, -65
        self.interval = playback_speed/fps
        self.next_frame = 0
        self.playback_speed = playback_speed
        self.saved_skill_frame = False
        self.last_frame = None
        self.fps = fps

    def __call__(self, model, data, state, row):
        if data.time + 1e-7 < self.next_frame:
            return
        from PIL import Image, ImageDraw
        if self.renderer is None:
            self.renderer = mujoco.Renderer(model, height=640, width=960)
        self.renderer.update_scene(data, camera=self.camera)
        frame = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 960, 64), fill='#152232')
        label = f"Go2 hybrid control  |  {state}"
        if row['skill']:
            label += f"  |  {row['skill']}"
        draw.text((18, 10), label, fill='white', font_size=20)
        draw.text((18, 38), f"t = {data.time:.1f} s   distance = {row['distance']:.2f} m   playback {self.playback_speed}x", fill='#abc7e0', font_size=15)
        self.writer.append_data(np.asarray(frame))
        self.last_frame = frame.copy()
        if state == 'SKILL' and not self.saved_skill_frame:
            frame.save(self.path.with_suffix('.png'))
            self.saved_skill_frame = True
        self.next_frame += self.interval

    def finish(self, summary):
        if self.last_frame is None:
            return
        from PIL import ImageDraw
        frame = self.last_frame.copy()
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 960, 64), fill='#174b36' if summary['success'] else '#702d29')
        draw.text((18, 10), f"Go2 hybrid control  |  {summary['reason'].upper()}", fill='white', font_size=20)
        draw.text((18, 38), f"Final error {summary['final_distance']:.3f} m  |  "
                  f"contact: {summary['collision']}  |  handoffs: {summary['returns_to_baseline']}",
                  fill='white', font_size=15)
        for _ in range(self.fps):
            self.writer.append_data(np.asarray(frame))
        frame.save(self.path.with_name(self.path.stem+'-final.png'))

    def close(self):
        self.writer.close()
        if self.renderer is not None:
            self.renderer.close()
