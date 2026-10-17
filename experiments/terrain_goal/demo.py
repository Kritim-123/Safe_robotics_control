"""Finite terrain/goal replay with live fresh-count and hold-state recording."""
import argparse,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.transfer import terrain_cases
from experiments.continual_snn.observations import SensorConfig
from .assessment import primary_config
from .goal_cases import goal_cases
from .loading_v2 import load_any
from .runner import execute


class CountMovie:
    def __init__(self,folder,case,selector):
        self.folder,self.case,self.selector=folder,case,selector
        self.next_frame=0.;self.renderer=self.writer=None;self.preview_saved=False

    def __call__(self,m,d,h):
        if d.time<self.next_frame:return
        import mujoco,imageio.v2 as imageio
        from PIL import Image,ImageDraw
        if self.renderer is None:
            self.renderer=mujoco.Renderer(m,height=480,width=900)
            self.writer=imageio.get_writer(str(self.folder/'recording.mp4'),fps=30)
        camera=mujoco.MjvCamera();camera.lookat[:]=[*d.qpos[:2],.22]
        camera.distance,camera.azimuth,camera.elevation=3.6,110+np.rad2deg(self.case.scenario.start_yaw),-28
        self.renderer.update_scene(d,camera=camera)
        frame=Image.fromarray(self.renderer.render());image=Image.new('RGB',(900,680),(17,22,29));image.paste(frame,(0,0))
        draw=ImageDraw.Draw(image);draw.rectangle((0,0,900,66),fill=(0,0,0))
        proposal=getattr(self.selector,'last_proposal',None) or {}
        error=float(np.linalg.norm(np.asarray(self.case.scenario.goal)-d.qpos[:2]))
        draw.text((12,7),f'{self.case.scenario.name} | t={d.time:.1f}s | mission={self.case.request_at(float(d.time))} | {h.current}/{h.mode}',fill='white')
        draw.text((12,26),f'Goal error {error:.3f} m | goal spikes {proposal.get("goal_count_sum",0)} | enter >=92 x2 | release <25 x2',fill='white')
        draw.text((12,45),f'Goal hold committed: {proposal.get("goal_latch_active",False)} | requested skill: {proposal.get("proposed_skill","settle")}',fill='white')
        draw.text((12,493),'Fresh 64-neuron packet (0 to 16 spikes per neuron); membranes and counts reset each decision',fill='white')
        counts=proposal.get('spike_counts') or [0]*64
        width=860/64
        for i,n in enumerate(counts):
            x=20+i*width;color=(64,166,242) if i<32 else (93,213,143)
            draw.rectangle((x,610-n*5,x+width-3,610),fill=color)
        draw.text((12,625),'Blue: task and clearance | Green: goal proximity | one persistent robot episode',fill='white')
        draw.text((12,645),'SNN chooses a skill; bounded joint PID and checked braking continue underneath',fill='white')
        self.writer.append_data(np.asarray(image));self.next_frame+=1/30
        if not self.preview_saved and d.time>5.:
            image.save(self.folder/'preview.png');self.preview_saved=True

    def close(self):
        if self.renderer is not None:self.renderer.close()
        if self.writer is not None:self.writer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=('steps','threshold','noisy_goal','refused_slope'),default='steps')
    p.add_argument('--model',type=Path,default=ROOT/'docs/agents/terrain_goal_v2/primary.npz')
    p.add_argument('--output',type=Path);p.add_argument('--viewer',action='store_true');p.add_argument('--video',action='store_true')
    a=p.parse_args();cfg=primary_config();sensor=None
    if a.scene=='noisy_goal':
        case=goal_cases(15100,1,'goal_replay')[0]
        cfg=replace(cfg,selector_position_noise=.02,selector_yaw_noise=.02);sensor=SensorConfig(.03,.015,.1,9917)
    else:
        index={'steps':3*6+3,'threshold':1*6+5,'refused_slope':5*6+5}[a.scene]
        case=terrain_cases(20261117)[index]
    model=load_any(a.model)
    if a.video and not hasattr(model,'last_proposal'):raise ValueError('Live count movie requires the V2 goal readout')
    output=(a.output or ROOT/'output/terrain_goal'/('demo-'+a.scene+'-'+str(time.time_ns()))).resolve()
    movie=CountMovie(output,case,model) if a.video else None
    try:
        result=execute(case,output,model=model,adaptation=cfg,sensor_config=sensor,viewer=a.viewer,
            full_trace=True,frame_callback=movie)
    finally:
        if movie is not None:movie.close()
    print(result,flush=True)


if __name__=='__main__':main()
