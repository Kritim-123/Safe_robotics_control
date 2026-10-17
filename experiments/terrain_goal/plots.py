"""Scientific figures from saved physics and packet records."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.continual_snn.models import ROOT
from .evidence import collect,rows_for,aggregate
from .loading_v2 import load_any

COLORS={'v1':'#e08055','v2':'#247bb4','rule':'#37935e'}


def finish(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'output/terrain_goal')
    p.add_argument('--output',type=Path,default=ROOT/'docs/figures')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    summary=collect(a.root,a.root/'evidence')
    fig,ax=plt.subplots(figsize=(9,4))
    def successes(phase,arm):return next(r['tasks'] for r in summary[phase] if r['arm']==arm)
    benchmark_values=[successes('baseline','v2'),successes('neural_only','new_selector_old_motor'),
        successes('original_benchmark','new'),successes('original_benchmark','rule')]
    ax.bar(range(4),benchmark_values,color=['#9099a3','#5ca6d5','#247bb4','#37935e'])
    ax.set_xticks(range(4),['Previous selector\noriginal gait','New selector\noriginal gait','New goal controller\nterrain profile','Rule goal controller\nterrain profile'])
    ax.set_ylim(0,66);ax.set_ylabel('Successful tasks out of 60')
    ax.set_title('Original terrain course: preserved baseline and matched controls')
    for i,n in enumerate(benchmark_values):ax.text(i,n+1,f'{n}/60',ha='center')
    ax.text(.01,.98,'Goal-controller rows additionally require supported arrival within 0.12 m.',transform=ax.transAxes,va='top',fontsize=9)
    finish(fig,a.output/'terrain-goal-original-benchmark.png')
    fig,ax=plt.subplots(figsize=(9,4))
    phases=('v2_clean','v2_mild','v2_severe');labels=('Clean','2 cm pose noise','5 cm pose noise')
    for j,arm in enumerate(('v1','v2','rule')):
        rates=[];texts=[]
        for phase in phases:
            row=next((r for r in summary[phase] if r['arm']==arm),None)
            rates.append(100*row['tasks']/row['trials'] if row else 0.)
            texts.append(f'{row["tasks"]}/{row["trials"]}' if row else 'pending')
        xs=np.arange(3)+(j-1)*.24
        ax.bar(xs,rates,width=.23,color=COLORS[arm],label={'v1':'Goal64','v2':'Goal64 hold','rule':'Rule'}[arm])
        for x,y,t in zip(xs,rates,texts):ax.text(x,y+1,t,ha='center',fontsize=8)
    ax.set_xticks(range(3),labels);ax.set_ylim(0,112);ax.set_ylabel('Strict goal success percent')
    ax.set_title('Fresh V2 confirmations with unchanged encoder and motor weights')
    ax.legend(ncol=3,loc='upper right');finish(fig,a.output/'terrain-goal-fresh-confirmation.png')
    model=load_any(ROOT/'docs/agents/terrain_goal_v2/primary.npz')
    f={n:0. for n in model.encoder.feature_names};f.update(goal_mode=1.,task_progress=1.,
        predicted_run_clearance=1.,predicted_jump_clearance=1.,footprint_clearance=1.)
    distance=np.linspace(0.,.28,281)
    counts=[sum(model.encoder.encode(dict(f,goal_forward=float(d)))['spike_counts'][32:]) for d in distance]
    fig,ax=plt.subplots(figsize=(8,3.7));ax.plot(distance,counts,color='#37935e',lw=2)
    ax.axhline(92,color='#247bb4',ls='--',label='Hold entry: 92 spikes')
    ax.axhline(25,color='#bd7849',ls=':',label='Release: below 25 spikes')
    ax.axvspan(.095,.16,color='#cbd4db',alpha=.5,label='Hold commitment separates the boundaries')
    ax.set_xlabel('Observed goal distance (m)');ax.set_ylabel('Total spikes in 32 goal neurons')
    ax.set_title('Monotonic proximity code with two fresh packets per boundary')
    ax.legend(fontsize=8);finish(fig,a.output/'terrain-goal-count-code.png')
    folder=a.root/'hysteresis-final-mild-v2'/'hysteresis_final_mild-goal-15100-000'
    if folder.exists():
        fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
        for arm in ('v1','v2'):
            ep=folder/arm
            if not (ep/'summary.json').exists():continue
            s=json.loads((ep/'summary.json').read_text());case=json.loads((ep/'scenario.json').read_text())
            packets=[r for x in (ep/'decisions.jsonl').read_text().splitlines() if (r:=json.loads(x))['task']=='progress']
            with np.load(ep/'trace.npz',allow_pickle=False) as z:values=z['values'];columns=z['columns'].tolist()
            goal=np.asarray(case['scenario']['goal']);xy=values[:,[columns.index('qpos_0'),columns.index('qpos_1')]]
            true_error=np.linalg.norm(xy-goal,axis=1)
            axes[0].plot(values[:,0],true_error,color=COLORS[arm],label={'v1':'Goal64','v2':'Goal64 hold'}[arm])
            times=[r['time'] for r in packets]
            axes[1].plot(times,[r['proposal'].get('goal_count_sum',0) for r in packets],color=COLORS[arm],lw=1)
            axes[2].step(times,[int(r['current']=='run')+(0.03 if arm=='v2' else 0.) for r in packets],where='post',color=COLORS[arm],label=arm.upper())
        axes[0].axhline(.12,color='black',ls=':',lw=1);axes[0].set_ylim(0,.35)
        axes[0].set_ylabel('True goal error (m)');axes[0].legend()
        axes[1].axhline(92,color='black',ls='--',lw=.8);axes[1].axhline(25,color='gray',ls=':',lw=.8)
        axes[1].set_ylabel('Goal spike count sum')
        axes[2].set_yticks([0,1],['Wait','Run']);axes[2].set_xlabel('Simulation time (s)')
        progress_end=max(r['end'] for r in case['requests'] if r['task']=='progress')
        axes[2].set_xlim(7,progress_end);axes[2].set_ylim(-.15,1.2)
        axes[0].set_title('Matched noisy goal: count hysteresis reduces repeated skill changes')
        finish(fig,a.output/'terrain-goal-noisy-handoffs.png')
    if summary['refinement_confirmation']:
        fig,ax=plt.subplots(figsize=(8.5,4))
        styles=(('v2','#247bb4','Prior readout'),('candidate','#8655c7','Request-refined'),('rule','#37935e','Rule'))
        for j,(arm,color,label) in enumerate(styles):
            rates=[];texts=[]
            for phase in ('refinement_confirmation','refinement_noise'):
                r=next((r for r in summary[phase] if r['arm']==arm),None)
                rates.append(100*r['tasks']/r['trials'] if r else 0.)
                texts.append(f'{r["tasks"]}/{r["trials"]}' if r else 'pending')
            xs=np.arange(2)+(j-1)*.24;ax.bar(xs,rates,width=.23,color=color,label=label)
            for x,y,t in zip(xs,rates,texts):ax.text(x,y+1,t,ha='center',fontsize=9)
        ax.set_xticks(range(2),['New clean scenes','New mild-noise scenes'])
        ax.set_ylim(0,112);ax.set_ylabel('Strict goal success percent')
        ax.set_title('More training states; unchanged encoder, goal readout and motors')
        ax.legend(ncol=3,loc='upper right');finish(fig,a.output/'terrain-goal-request-refinement.png')
    selected=a.root/'selected-passage-demo.json'
    if selected.exists():
        import imageio.v2 as imageio
        scene=json.loads(selected.read_text());fig,axes=plt.subplots(1,2,figsize=(13,6))
        for ax,name,arm,title in zip(axes,('demo-passage-prior-v2','demo-passage-refined-v3'),
                                   ('v2','candidate'),('Prior readout','Request-refined readout')):
            ep=a.root/name/scene['case']/arm
            reader=imageio.get_reader(ep/'recording.mp4');frame=reader.get_data(900);reader.close()
            with np.load(ep/'trace.npz') as z:values=z['values'];columns=z['columns'].tolist()
            i=np.argmin(abs(values[:,0]-30.));xy=values[i,[columns.index('qpos_0'),columns.index('qpos_1')]]
            goal=np.asarray(json.loads((ep/'scenario.json').read_text())['scenario']['goal'])
            ax.imshow(frame);ax.axis('off');ax.set_title(f'{title}: goal error {np.linalg.norm(xy-goal):.2f} m',fontsize=13)
        fig.suptitle('Same passage at about 30 s; unchanged encoder and motor controller',fontsize=15)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(a.output/'terrain-goal-passage-example.png',dpi=170,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
