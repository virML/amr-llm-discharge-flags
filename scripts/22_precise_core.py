"""Unrounded re-derivation of Table 3/4 statistics using the identical bootstrap procedures and seeds
as 09_modeling.py (CIs: seed 42, 1000 patient-cluster resamples) and 10_delta_ci.py (paired deltas: seed 123)."""
import json, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ll=json.load(open('data/processed/model_outputs/llm/predictions.json')); rb=json.load(open('data/processed/model_outputs/predictions.json'))
def groups(pid):
    d={}
    for i,p in enumerate(pid): d.setdefault(p,[]).append(i)
    return d, np.array(list(d.keys()),dtype=object)
def ci(y,s,pid,fn,seed=42,n=1000):
    rng=np.random.RandomState(seed); d,keys=groups(pid); v=[]
    for _ in range(n):
        idx=np.concatenate([d[k] for k in keys[rng.randint(0,len(keys),len(keys))]])
        if len(np.unique(y[idx]))<2: continue
        try: v.append(fn(y[idx],s[idx]))
        except Exception: continue
    return float(np.percentile(v,2.5)), float(np.percentile(v,97.5))
def delta(y,pa,pb,pid,seed=123,n=1000):
    rng=np.random.RandomState(seed); d,keys=groups(pid); A=[];P=[]
    for _ in range(n):
        idx=np.concatenate([d[k] for k in keys[rng.randint(0,len(keys),len(keys))]])
        if len(np.unique(y[idx]))<2: continue
        A.append(roc_auc_score(y[idx],pb[idx])-roc_auc_score(y[idx],pa[idx])); P.append(average_precision_score(y[idx],pb[idx])-average_precision_score(y[idx],pa[idx]))
    return dict(dAUROC=float(np.mean(A)),dAUROC_lo=float(np.percentile(A,2.5)),dAUROC_hi=float(np.percentile(A,97.5)),dAUPRC=float(np.mean(P)),dAUPRC_lo=float(np.percentile(P,2.5)),dAUPRC_hi=float(np.percentile(P,97.5)))
out={'metrics':{},'delta_llm':{},'delta_rb':{},'llm_vs_rb':{}}
for t in ['MRSA','ESBL','CRE']:
  for split in ['split_temporal','split_random']:
    for fs in ['A','B','Baseline0_organism_only','Baseline1_organism_specimen_ward_onset']:
      for m in ['logreg','rf','xgb']:
        for run,src in [('llm',ll),('rb',rb)]:
          k=f'{t}__{split}__{fs}__{m}'
          if k not in src: continue
          if run=='rb' and fs!='B': continue
          d=src[k]; y=np.array(d['y_test']); p=np.array(d['y_proba']); pid=np.array(d['study_id'])
          a_lo,a_hi=ci(y,p,pid,roc_auc_score); p_lo,p_hi=ci(y,p,pid,average_precision_score)
          out['metrics'][f'{run}__{k}']=dict(AUROC=float(roc_auc_score(y,p)),AUROC_lo=a_lo,AUROC_hi=a_hi,AUPRC=float(average_precision_score(y,p)),AUPRC_lo=p_lo,AUPRC_hi=p_hi,Brier=float(brier_score_loss(y,p)))
    for m in ['logreg','rf','xgb']:
      for run,src in [('llm',ll),('rb',rb)]:
        ka=f'{t}__{split}__A__{m}'; kb=f'{t}__{split}__B__{m}'
        y=np.array(src[ka]['y_test']); out[f'delta_{run}'][f'{t}__{split}__{m}']=delta(y,np.array(src[ka]['y_proba']),np.array(src[kb]['y_proba']),np.array(src[ka]['study_id']))
      kb=f'{t}__{split}__B__{m}'; y=np.array(ll[kb]['y_test'])
      out['llm_vs_rb'][f'{t}__{split}__{m}']=delta(y,np.array(rb[kb]['y_proba']),np.array(ll[kb]['y_proba']),np.array(ll[kb]['study_id']))
json.dump(out,open('tables/llm_precise/core_precise.json','w'),indent=1)
# sanity vs rounded tables
import pandas as pd
pl=pd.read_csv('tables/llm/table4_model_performance_full.csv'); bad=0
for _,r in pl.iterrows():
    x=out['metrics'][f"llm__{r.target}__{r.split}__{r.feature_set}__{r.model}"]
    for a,b in [('AUROC','AUROC'),('AUROC_lo','AUROC_CI_lo'),('AUROC_hi','AUROC_CI_hi'),('AUPRC','AUPRC'),('AUPRC_lo','AUPRC_CI_lo'),('AUPRC_hi','AUPRC_CI_hi')]:
        if abs(round(x[a],4)-r[b])>0.00011: bad+=1; print('MISMATCH',r.target,r.split,r.feature_set,r.model,a,x[a],r[b])
dl=pd.read_csv('tables/llm/table4_paired_delta_ci.csv')
for _,r in dl.iterrows():
    x=out['delta_llm'][f'{r.target}__{r.split}__{r.model}']
    for a,b in [('dAUROC','AUROC_delta_mean'),('dAUROC_lo','AUROC_delta_CI_lo'),('dAUROC_hi','AUROC_delta_CI_hi'),('dAUPRC','AUPRC_delta_mean')]:
        if abs(round(x[a],4)-r[b])>0.00011: bad+=1; print('MISMATCH delta',r.target,r.split,r.model,a,x[a],r[b])
print('sanity mismatches:',bad)
