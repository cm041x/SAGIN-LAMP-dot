"""Plot lifecycle damping distributions from saved pilot JSON, not live state."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/lamp_prototype'
statuses=('standby','active','returning','waiting_charge','charging')
variants=('lamp_plain','lamp_lifecycle','lamp_lifecycle_ecc')
fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,variant in zip(axes,variants):
    history=json.loads((OUT/'training'/variant/'history.json').read_text())
    values=[[v for episode in history for v in episode['metrics']['damping_by_status_samples'][s]] for s in statuses]
    present=[(i+1,x) for i,x in enumerate(values) if x]
    if present: ax.boxplot([x for _,x in present],positions=[i for i,_ in present],showfliers=False)
    ax.set_xticks(range(1,len(statuses)+1),statuses,rotation=45,ha='right');ax.set_title(variant)
    ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Mean channel damping per UAV / step')
fig.tight_layout();fig.savefig(OUT/'status_damping_by_status.png',dpi=160);plt.close(fig)
