"""Point estimates for all model comparisons = OBSERVED test-set difference (metric of model 2 minus metric of model 1),
with the unchanged percentile CIs from the paired patient-cluster bootstrap. Earlier tables used the bootstrap mean as the
point estimate; this script derives observed differences from the full-precision metrics and keeps the bootstrap mean for
reference. Outputs tables/obs/."""
import json, pandas as pd, numpy as np
core=json.load(open('tables/llm_precise/core_precise.json'))
M=core['metrics']; out={'metrics':M}
bad=[]
def chk(name,obs,lo,hi):
    if not (lo-1e-12<=obs<=hi+1e-12): bad.append((name,obs,lo,hi))
for k in ['delta_llm','delta_rb','llm_vs_rb']:
    out[k]={}
    for key,d in core[k].items():
        t,sp,m=key.split('__')
        if k=='llm_vs_rb': a=M[f'rb__{t}__{sp}__B__{m}']; b=M[f'llm__{t}__{sp}__B__{m}']
        else:
            a=M[f'llm__{t}__{sp}__A__{m}']; b=M[f"{'llm' if k=='delta_llm' else 'rb'}__{t}__{sp}__B__{m}"]
        e=dict(d); e['dAUROC_bootmean']=d['dAUROC']; e['dAUPRC_bootmean']=d['dAUPRC']
        e['dAUROC']=b['AUROC']-a['AUROC']; e['dAUPRC']=b['AUPRC']-a['AUPRC']
        chk(f'{k}:{key}:AUROC',e['dAUROC'],e['dAUROC_lo'],e['dAUROC_hi']); chk(f'{k}:{key}:AUPRC',e['dAUPRC'],e['dAUPRC_lo'],e['dAUPRC_hi'])
        out[k][key]=e
json.dump(out,open('tables/obs/core_obs.json','w'),indent=1)
# multiplicity tables (5000-rep CIs and P values; observed point estimates)
for src,dst,run in [('tables/llm/table4_multiple_testing_correction.csv','tables/obs/mt_llm.csv','llm'),('tables/table4_multiple_testing_correction.csv','tables/obs/mt_rb.csv','rb')]:
    mt=pd.read_csv(src); obs=[]
    for _,r in mt.iterrows():
        a=M[f'llm__{r.target}__{r.split}__A__{r.model}']; b=M[f'{run}__{r.target}__{r.split}__B__{r.model}']
        o=b[r.metric]-a[r.metric]; obs.append(o); chk(f'mt_{run}:{r.target}:{r.split}:{r.model}:{r.metric}',o,r.ci_lo-5e-5,r.ci_hi+5e-5)
    mt['delta_bootmean']=mt['delta_mean']; mt['delta_obs']=obs; mt.to_csv(dst,index=False)
# ablation (observed = AUROC_B - AUROC_A already in table)
ab=pd.read_csv('tables/llm_precise/table_llm_ablation.csv')
ab['dAUROC_bootmean']=ab.dAUROC; ab['dAUPRC_bootmean']=ab.dAUPRC; ab['dAUROC']=ab.AUROC_B-ab.AUROC_A; ab['dAUPRC']=ab.AUPRC_B-ab.AUPRC_A
for _,r in ab.iterrows(): chk(f'abl:{r.target}:{r.model}:{r.feature_set}:AUROC',r.dAUROC,r.dAUROC_lo,r.dAUROC_hi); chk(f'abl:{r.target}:{r.model}:{r.feature_set}:AUPRC',r.dAUPRC,r.dAUPRC_lo,r.dAUPRC_hi)
ab.to_csv('tables/obs/ablation_obs.csv',index=False)
for src,dst in [('tables/llm_precise/table_note_present_sensitivity.csv','tables/obs/note_present_llm_obs.csv'),('tables/rb_precise/table_note_present_sensitivity.csv','tables/obs/note_present_rb_obs.csv')]:
    n=pd.read_csv(src); n['AUROC_delta_bootmean']=n.AUROC_delta; n['AUPRC_delta_bootmean']=n.AUPRC_delta
    n['AUROC_delta']=n.AUROC_B-n.AUROC_A; n['AUPRC_delta']=n.AUPRC_B-n.AUPRC_A
    for _,r in n.iterrows(): chk(f'np:{dst}:{r.target}:AUROC',r.AUROC_delta,r.AUROC_delta_CI_lo,r.AUROC_delta_CI_hi); chk(f'np:{dst}:{r.target}:AUPRC',r.AUPRC_delta,r.AUPRC_delta_CI_lo,r.AUPRC_delta_CI_hi)
    n.to_csv(dst,index=False)
bd=pd.read_csv('tables/llm/table4b_baseline_delta_ci.csv'); obs=[]
for _,r in bd.iterrows():
    bk={'organism_only':'Baseline0_organism_only','organism_specimen_ward_onset':'Baseline1_organism_specimen_ward_onset'}[r.baseline]
    a=M[f'llm__{r.target}__split_temporal__A__{r.model_family}']['AUROC']; b=M[f'llm__{r.target}__split_temporal__{bk}__logreg']['AUROC']
    obs.append(a-b); chk(f'base:{r.target}:{r.baseline}',a-b,r.AUROC_delta_CI_lo,r.AUROC_delta_CI_hi)
bd['AUROC_delta_bootmean']=bd.AUROC_delta_full_minus_baseline; bd['AUROC_delta_full_minus_baseline']=obs; bd['AUROC_full_precise']=[M[f'llm__{t}__split_temporal__A__{m}']['AUROC'] for t,m in zip(bd.target,bd.model_family)]
bd['AUROC_baseline_precise']=[M[f"llm__{t}__split_temporal__{ {'organism_only':'Baseline0_organism_only','organism_specimen_ward_onset':'Baseline1_organism_specimen_ward_onset'}[bl] }__logreg"]['AUROC'] for t,bl in zip(bd.target,bd.baseline)]
bd.to_csv('tables/obs/baseline_obs.csv',index=False)
print('observed outside CI:',len(bad)); [print(' ',b) for b in bad]
