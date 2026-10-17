"""Generate compact tables/figures from immutable results; never fit policies."""
import argparse
import csv
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from .models import ROOT
from .study import aggregate,save

BASE=ROOT/'output/continual_snn'
FINAL_BATCHES={f'final-{name}-v1' for name in (
    'main','stress','confirmation','noise','severe_noise','heading','terrain','stale')}


def failure_signals(s):
    values=[]
    if not s['episode_completed']:values.append(s['reason'])
    if s['progress']<s['required_progress']-.01:values.append('insufficient progress')
    if s['completed_jumps']<s['required_jumps']:values.append('missing verified hop')
    if s.get('maximum_jumps') is not None and s['completed_jumps']>s['maximum_jumps']:values.append('excess hops')
    if s['hold_violation_packets']:values.append('hold not stable after grace')
    if s['failed_jumps']:values.append('motor hop failure')
    return values


def collect():
    records=[];batches=[]
    for path in sorted(BASE.glob('*/results.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        for row in rows:records.append(dict(batch=path.parent.name,arm=row['arm'],**row['summary']))
        batches.append(dict(batch=path.parent.name,**aggregate(rows)))
    return records,batches


def write_csv(path,rows,fields):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in rows:writer.writerow({k:row.get(k) for k in fields})


def generate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records,batches=collect();save(output/'cohorts.json',batches)
    fields=['batch','arm','case','family','seed','task_fulfilled','episode_completed','reason','progress',
        'required_progress','completed_jumps','required_jumps','failed_jumps','cancelled_jumps',
        'hold_violation_packets','collision','fall','minimum_clearance','max_tilt','max_torque',
        'max_torque_limit_ratio','switches','elapsed','neural_evaluations','model_checksum']
    write_csv(output/'episodes.csv',records,fields)
    failures=[dict(r,failure_signals='; '.join(failure_signals(r))) for r in records if not r['task_fulfilled']]
    write_csv(output/'failures.csv',failures,fields+['failure_signals'])
    summaries=[]
    for cohort in batches:
        for group in cohort['groups']:summaries.append(dict(batch=cohort['batch'],**group))
    write_csv(output/'cohorts.csv',summaries,['batch','arm','trials','tasks','completed','collisions','falls',
        'hold_violations','mean_progress','total_switches','total_hops','max_torque'])
    final=[r for r in records if r['batch'] in FINAL_BATCHES]
    coverage=[r for r in records if r['batch'] in {'coverage-v2-confirm-'+p for p in ('stress','regular','noise')}]
    save(output/'totals.json',dict(all_episodes=len(records),final_episodes=len(final),
        all_cases=len({(r['batch'],r['case']) for r in records}),
        final_unique_scene_cases=len({r['case'] for r in final}),
        all_contacts=sum(r['collision'] for r in records),all_falls=sum(r['fall'] for r in records),
        final_contacts=sum(r['collision'] for r in final),final_falls=sum(r['fall'] for r in final),
        v2_fresh_confirmation_episodes=len(coverage),
        v2_unique_named_confirmation_cases=len({r['case'] for r in coverage}),
        counts_are_episodes_not_independent_frames=True))
    plot_results(output,batches);plot_coverage(output,batches)
    return records,batches


def plot_results(output,batches):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    choices=['old_repeat','new_once','new_repeat','prototype80','sensory64','analog24','rule','always_run']
    lookup={b['batch']:b for b in batches}
    names=[n for n in ('main','stress','confirmation','noise','severe_noise','heading','terrain','stale')
           if 'final-'+n+'-v1' in lookup]
    if not names:return
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    colors=dict(old_repeat='#9b9b9b',new_once='#b28abd',new_repeat='#167fc0',prototype80='#23976b',
        sensory64='#e89c32',analog24='#bc5252',rule='#4e526b',always_run='#b2a17d')
    for ax,name in zip(axes.flat,names):
        groups={r['arm']:r for r in lookup['final-'+name+'-v1']['groups']}
        selected=[n for n in choices if n in groups]
        ax.bar(range(len(selected)),[100*groups[n]['tasks']/groups[n]['trials'] for n in selected],
            color=[colors[n] for n in selected])
        for i,n in enumerate(selected):
            r=groups[n];ax.text(i,min(101,100*r['tasks']/r['trials']+2),f'{r["tasks"]}/{r["trials"]}',
                ha='center',fontsize=8,rotation=90)
        ax.set_xticks(range(len(selected)),selected,rotation=65,ha='right',fontsize=8)
        ax.set_ylim(0,125);ax.set_yticks([0,25,50,75,100]);ax.set_title(name.replace('_',' '))
        ax.set_ylabel('Tasks fulfilled (%)');ax.grid(axis='y',alpha=.2)
    for ax in list(axes.flat)[len(names):]:ax.set_visible(False)
    fig.suptitle('Frozen SNN evaluation | task success differs from safe episode completion',fontsize=14)
    fig.savefig(output/'outcomes.png',dpi=160);plt.close(fig)


def plot_coverage(output,batches):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup={b['batch']:b for b in batches}
    names=[p for p in ('stress','regular','noise') if 'coverage-v2-confirm-'+p in lookup]
    if not names:return
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    order=['v1_readout80','v1_compact16','v1_prototype80','v2_selected','rule']
    labels=['old 80 readout','old 16 bank','old 80 bank','updated 80 bank','rule']
    for ax,phase in zip(axes,names):
        groups={r['arm']:r for r in lookup['coverage-v2-confirm-'+phase]['groups']}
        control=lookup.get('coverage-v2-control-'+phase,{}).get('groups',[])
        groups.update({r['arm']:r for r in control})
        selected=[i for i,n in enumerate(order) if n in groups]
        ax.bar(range(len(selected)),[100*groups[order[i]]['tasks']/groups[order[i]]['trials'] for i in selected],
            color=[('#23976b' if order[i]=='v2_selected' else '#8d99ad') for i in selected])
        for x,i in enumerate(selected):
            r=groups[order[i]];ax.text(x,100*r['tasks']/r['trials']+2,f'{r["tasks"]}/{r["trials"]}',
                ha='center',fontsize=9)
        ax.set_xticks(range(len(selected)),[labels[i] for i in selected],rotation=45,ha='right')
        ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.set_title(phase)
        ax.set_ylabel('Tasks fulfilled (%)');ax.grid(axis='y',alpha=.2)
    for ax in axes[len(names):]:ax.set_visible(False)
    fig.suptitle('Fresh V2 confirmation | updated association data, unchanged encoder and motor skills')
    fig.savefig(output/'coverage-v2.png',dpi=160);plt.close(fig)


def timeline(episode,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    episode=Path(episode);rows=[json.loads(s) for s in (episode/'decisions.jsonl').read_text().splitlines()]
    with np.load(episode/'trace.npz') as bundle:v=bundle['values'];columns=bundle['columns'].tolist()
    mapping=dict(wait=0,run=1,jump=2)
    t=[r['time'] for r in rows]
    fig,axes=plt.subplots(4,1,figsize=(13,9),sharex=True,layout='constrained')
    axes[0].step(t,[mapping[r['desired']] for r in rows],where='post',label='desired',alpha=.7)
    axes[0].step(t,[mapping[r['current']] for r in rows],where='post',label='executing',ls='--')
    axes[0].set_yticks([0,1,2],['wait','run','jump']);axes[0].legend(ncol=2)
    axes[0].set_title('Fresh SNN packets, stable handoffs, persistent MuJoCo physics')
    height,=axes[1].plot(v[:,0],v[:,columns.index('qpos_2')],label='base height',color='#167fc0')
    speed_axis=axes[1].twinx()
    speed,=speed_axis.plot(v[:,0],np.linalg.norm(v[:,[columns.index('qvel_'+str(i)) for i in range(3)]],axis=1),
        label='base speed',color='#23976b',alpha=.8)
    axes[1].set_ylabel('Height (m)');speed_axis.set_ylabel('Speed (m/s)')
    axes[1].legend(handles=[height,speed],ncol=2);axes[1].grid(alpha=.2)
    counts=np.asarray([r['proposal']['spike_counts'] for r in rows],dtype=float)
    heat=axes[2].imshow(counts.T,aspect='auto',origin='lower',extent=(t[0],t[-1],-.5,counts.shape[1]-.5),
        vmin=0,vmax=16,cmap='viridis',interpolation='nearest')
    if counts.shape[1]>64:axes[2].axhline(63.5,color='white',ls=':',lw=1.)
    fig.colorbar(heat,ax=axes[2],label='Spikes/packet',ticks=[0,4,8,12,16])
    axes[2].set_ylabel('Output neuron ID');axes[2].set_title('Spike count 0–16 per independent 100 ms observation packet',fontsize=10)
    axes[3].plot(v[:,0],np.max(np.abs(v[:,[columns.index('torque_'+str(i)) for i in range(12)]]),axis=1))
    axes[3].axhline(28,color='r',ls=':',label='hop bound');axes[3].set_ylabel('Peak |torque| (N m)')
    axes[3].set_xlabel('Simulation time (s)');axes[3].grid(alpha=.2)
    fig.savefig(output,dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeline',type=Path)
    args=parser.parse_args();generate(args.output)
    if args.timeline:timeline(args.timeline,args.output/'handoffs.png')
