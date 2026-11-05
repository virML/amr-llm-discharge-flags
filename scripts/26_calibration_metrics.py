"""Calibration intercept (calibration-in-the-large, slope fixed at 1) and calibration slope, with patient-cluster bootstrap
95% CIs (seed 42, 1000 resamples), for the headline Model A and Model B (LLM), before and after isotonic recalibration
(temporal test set). Output tables/supp/calibration_metrics.csv"""
import json, numpy as np, pandas as pd
rc=json.load(open('data/processed/model_outputs/llm/recalibration_predictions.json'))
pr=json.load(open('data/processed/model_outputs/llm/predictions.json'))
EPS=1e-6
def logit(p): p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p))
def fit(y,x,offset=False):
    # Newton-Raphson logistic regression; offset=True: y ~ 1 + offset(x) -> returns intercept; else y ~ 1 + x -> (a,b)
    y=np.asarray(y,float); X=np.ones((len(y),1)) if offset else np.column_stack([np.ones(len(y)),x]); off=x if offset else 0.0
    beta=np.zeros(X.shape[1])
    for _ in range(100):
        eta=X@beta+off; mu=1/(1+np.exp(-eta)); W=mu*(1-mu)+1e-12
        g=X.T@(y-mu); H=(X*W[:,None]).T@X; step=np.linalg.solve(H,g); beta+=step
        if np.max(np.abs(step))<1e-10: break
    return beta
def metrics(y,p):
    lp=logit(p); a=fit(y,lp,offset=True)[0]; b=fit(y,lp)[1]; return a,b,float(np.mean(y))/float(np.mean(p))
rows=[]
for t,m in [('MRSA','rf'),('ESBL','xgb'),('CRE','xgb')]:
    for fs in ['A','B']:
        d=rc[f'{t}__{fs}__{m}']; y=np.array(d['y_test']); pid=np.array(pr[f'{t}__split_temporal__{fs}__{m}']['study_id'])
        assert np.array_equal(y,np.array(pr[f'{t}__split_temporal__{fs}__{m}']['y_test']))
        groups={}; [groups.setdefault(p_,[]).append(i) for i,p_ in enumerate(pid)]; keys=np.array(list(groups.keys()),dtype=object)
        for stage,pk in [('before','p_uncal'),('after','p_cal')]:
            p=np.array(d[pk]); a,b,oe=metrics(y,p)
            rng=np.random.RandomState(42); A=[];Bs=[];O=[]
            for _ in range(1000):
                idx=np.concatenate([groups[k] for k in keys[rng.randint(0,len(keys),len(keys))]])
                if len(np.unique(y[idx]))<2: continue
                aa,bb,oo=metrics(y[idx],p[idx]); A.append(aa);Bs.append(bb);O.append(oo)
            q=lambda v:(float(np.percentile(v,2.5)),float(np.percentile(v,97.5)))
            rows.append(dict(target=t,model=m,feature_set=fs,stage=stage,intercept=a,intercept_lo=q(A)[0],intercept_hi=q(A)[1],slope=b,slope_lo=q(Bs)[0],slope_hi=q(Bs)[1],OE=oe,OE_lo=q(O)[0],OE_hi=q(O)[1]))
out=pd.DataFrame(rows); out.to_csv('tables/supp/calibration_metrics.csv',index=False); print(out.round(3).to_string(index=False))
