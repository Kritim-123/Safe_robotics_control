"""Fresh obstacle tests and a video of the frozen wait/run/jump SNN."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from hybrid.arena import Scenario, Obstacle
from hybrid.behavior_dataset import observation_features
from hybrid.skill_lif import SkillLIF, SKILLS, DEFAULT_MODEL
from skills.dynamic import execute_dynamic_skill

NAMES = ('crate_lane', 'low_bar', 'sliding_gate', 'approaching_crate',
         'jump_near_posts', 'jump_slippery')


def scene(name, seed):
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(1.0, 1.2))
    jump = name.startswith('jump_')
    friction = .8
    if name == 'crate_lane':
        obstacles = tuple(Obstacle(f'crate{i}', (x, side*1.0), shape='box',
                          half_extents=(.3,.15), height=.5) for i,side in enumerate((-1,1)))
    elif name == 'low_bar':
        obstacles = (Obstacle('low_bar', (x,0), shape='box', half_extents=(.08,.65), height=.08),)
    elif name == 'sliding_gate':
        obstacles = (Obstacle('gate', (x,-1.5), shape='box', half_extents=(.12,.3),
                              height=.6, velocity=(0,.25), start_after=2, stop_after=8),)
    elif name == 'approaching_crate':
        obstacles = (Obstacle('approach', (2.2,0), shape='box', half_extents=(.2,.2),
                              height=.5, velocity=(-.25,0), start_after=2, stop_after=6),)
    elif name == 'jump_near_posts':
        obstacles = (Obstacle('post_left', (.4,1.05), .12), Obstacle('post_right', (.4,-1.05), .12))
    elif name == 'jump_slippery':
        obstacles = (Obstacle('distant_crate', (2.5,.8), shape='box', half_extents=(.2,.2), height=.4),)
        friction = float(rng.uniform(.25,.35))
    elif name == 'jump_control':
        obstacles = ()
    else:
        raise ValueError(name)
    return Scenario(name, goal=(3,0), bounds=(-3,6,-3,3), obstacles=obstacles, friction=friction), jump


class Movie:
    def __init__(self, writer, name, decision):
        import mujoco
        self.writer, self.name, self.decision = writer, name, decision
        self.renderer = None
        self.next_frame = 0.
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [.6,0,.2]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 3.8, 120, -27
        self.last = None

    def __call__(self, model, data, skill):
        if data.time < self.next_frame: return
        import mujoco
        from PIL import Image, ImageDraw
        if self.renderer is None: self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.renderer.update_scene(data, camera=self.camera)
        frame = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0,0,640,76), fill='#142331')
        draw.text((12,8), f'{self.name} | {skill} | t={data.time:.2f}s', fill='white', font_size=19)
        counts = self.decision.get('counts')
        draw.text((12,36), f'Output spikes [wait, run, jump]: {counts if counts else "observing"}', fill='#a6d2ed', font_size=16)
        draw.text((12,58), 'Frozen SNN | real-time playback | jump is in-place', fill='#a6d2ed', font_size=13)
        self.last = np.asarray(frame)
        self.writer.append_data(self.last)
        self.next_frame += 1/30

    def finish(self, summary, path):
        from PIL import Image, ImageDraw
        if self.last is not None:
            frame = Image.fromarray(self.last)
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0,400,640,480), fill='#142331')
            draw.text((12,411), f"Result: {summary['reason']} | task fulfilled: {summary['task_success']}", fill='white', font_size=17)
            draw.text((12,441), f"Flight {summary['longest_airtime']:.3f}s | progress {summary['displacement'][0]:.2f}m", fill='white', font_size=17)
            frame.save(path)
            for _ in range(30): self.writer.append_data(np.asarray(frame))
        if self.renderer is not None: self.renderer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/skill_lif/showcase'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[5100,5101,5102])
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--viewer',action='store_true',help='Watch each scenario live in MuJoCo; close the window to stop')
    parser.add_argument('--scenarios',nargs='+',choices=(*NAMES,'jump_control'),default=list(NAMES))
    args=parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds): parser.error('Unique seeds required')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    policy=SkillLIF(DEFAULT_MODEL)
    writer=None
    if args.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(str(output/'showcase.mp4'),fps=30,codec='libx264',quality=7,macro_block_size=16)
    summaries=[]
    try:
        for name in args.scenarios:
            for seed in args.seeds:
                spec,jump=scene(name,seed)
                decision={}
                def choose(observation,allowed):
                    features=observation_features(spec,observation,jump)
                    choice,reason,counts,diag=policy.choose(features,[allowed[s] for s in SKILLS])
                    decision.update(features=features.tolist(),allowed=allowed,counts=counts.tolist(),
                                    reason=reason,output_spikes=diag['output_spikes'].tolist())
                    return SKILLS[choice] if choice is not None else None
                movie=Movie(writer,name,decision) if writer is not None and seed==args.seeds[0] else None
                history,summary=execute_dynamic_skill(scenario=spec,seed=seed,selector=choose,
                                                     frame_callback=movie,viewer=args.viewer)
                summary.update(scenario=name,jump_requested=jump,decision=decision,
                    task_success=bool(summary['success'] and (summary['skill']=='jump' and summary['all_feet_airborne'] if jump else summary['skill']=='run' and summary['displacement'][0]>.1)))
                folder=output/f'{name}-{seed}';folder.mkdir(exist_ok=True)
                if movie: movie.finish(summary,folder/'result.png')
                for filename,value in [('summary',summary),('scenario',spec.to_dict())]:
                    (folder/f'{filename}.json').write_text(json.dumps(value,indent=2,allow_nan=False))
                with (folder/'trajectory.csv').open('w',newline='') as stream:
                    csvwriter=csv.DictWriter(stream,fieldnames=list(history[0]));csvwriter.writeheader();csvwriter.writerows(history)
                summaries.append(summary)
                print(json.dumps({k:summary[k] for k in ('scenario','seed','skill','success','task_success','reason')}),flush=True)
                if summary['reason'] == 'viewer closed':
                    return
    finally:
        if writer: writer.close()
    (output/'evaluation.json').write_text(json.dumps(dict(seeds=args.seeds,model_sha256=hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest(),results=summaries),indent=2,allow_nan=False))
    fields=['scenario','seed','skill','success','task_success','reason','longest_airtime','min_clearance','max_torque','collision','fall']
    with (output/'results.csv').open('w',newline='') as stream:
        csvwriter=csv.DictWriter(stream,fieldnames=fields);csvwriter.writeheader();csvwriter.writerows({k:s[k] for k in fields} for s in summaries)


if __name__=='__main__': main()
