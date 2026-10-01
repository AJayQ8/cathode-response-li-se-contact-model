"""Separate two already published panels; plot saved values only, no model calls."""
from pathlib import Path
import csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
output=root/'figures/regenerated'
output.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
 'axes.titlesize':10,'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
 'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
 'axes.axisbelow':True,'savefig.facecolor':'white'})
source=root/'data/load_sharing_8_rows.csv'
with source.open() as f: rows=list(csv.DictReader(f))
assert len(rows)==8
fig,ax=plt.subplots(figsize=(6.4,3.65),layout='constrained')
fig.set_constrained_layout_pads(w_pad=.12,h_pad=.10)
for law,col,marker,shift,label in [('local','#246eaa','o',-.08,'Local nominal loading'),('common','#b85d22','s',.08,'Common active-area traction')]:
    for r in [r for r in rows if r['law']==law]:
        j=(0 if float(r['Q'])==.3 else 2)+(r['stage']!='end_discharge')
        ax.errorbar(j+shift,float(r['additional_feedback_pp']),yerr=float(r['numerical_diagnostic_error_pp']),fmt=marker,color=col,ms=5,capsize=3,label=label if j==0 else None)
ax.axhline(0,color='.55',lw=.8)
ax.set(xticks=range(4),xticklabels=['0.30\nCurrent stop','0.30\nAfter rest','0.60\nCurrent stop','0.60\nAfter rest'],ylabel='Increment in P/N history contrast (pp)',xlabel='Discharge charge (mAh cm$^{-2}$) and readout',xlim=(-.45,3.45))
ax.legend(frameon=False,fontsize=9,loc='upper left')
fig.savefig(output/'figure3_feedback.pdf',metadata={'CreationDate':None,'ModDate':None})
fig.savefig(output/'figure3_feedback.png',dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(6.4,3.6),layout='constrained')
fig.set_constrained_layout_pads(w_pad=.12,h_pad=.10)
# These are the frozen bar values from the previous figure, not a recomputation.
bad=[.0487804878,.0487804878,1.9512195122,1.9512195122]; good=[2.,2.,0.,0.]
ax.bar([i-.17 for i in range(4)],bad,width=.32,color='#a6aab0',label='Inverse-contact rule')
ax.bar([i+.17 for i in range(4)],good,width=.32,color='#246eaa',label='Conducting-contact limit')
ax.set(xticks=range(4),xticklabels=['Contact 1','Contact 2','Gap 1','Gap 2'],ylabel='Nominal patch current / applied current',ylim=(0,2.7))
ax.legend(frameon=False,fontsize=9,loc='upper center',ncol=2)
fig.savefig(output/'figureS1_current_routing.pdf',metadata={'CreationDate':None,'ModDate':None})
fig.savefig(output/'figureS1_current_routing.png',dpi=180)
plt.close(fig)
(output/'spatial_figure_provenance.json').write_text(json.dumps({'scope':'Replot saved panels separately; no scientific computation','main_figure':'figures/regenerated/figure3_feedback.pdf','input':str(source.relative_to(root)),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'supplementary_figure':'figures/regenerated/figureS1_current_routing.pdf','routing_bar_values':{'inverse':bad,'conducting':good},'routing_source':'../manuscript_original_format_20260928/build_figures.py, former Figure 3a'},indent=2)+'\n')
print('Rendered main Figure 3 and SI Figure S1 from unchanged values.')
