"""CRE (and ESBL) ablation on the FINAL pipeline: which LLM-extracted flags carry the Model B gain?
History-only flags (pre-admission concepts) vs hospital-course flags (possibly post-culture).
Same preprocessing / models / inner split as 09_modeling.py; paired patient-cluster bootstrap vs Model A."""
import json, numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
RS=42; NB=1000
struct=pd.read_parquet('data/processed/splits.parquet').reset_index(drop=True)
nlp=pd.read_parquet('data/processed/nlp_features_llm.parquet').reset_index(drop=True)
assert (struct.study_id.values==nlp.study_id.values).all()
cols=[c for c in nlp.columns if c not in ('study_id','isolate_id','testing_date')]
data=pd.concat([struct,nlp[cols]],axis=1)
NUM_A=['age_years','admission_to_culture_hours','prior_isolate_count','days_since_last_isolate','percent_probability_num']
CAT_A=['organism_base','specimen_type_clean','infection_site_clean','gender_clean','ward_clean','dept_clean','onset_bucket','id_confidence_clean']
BIN_A=['is_pediatric_flag','prior_resistant_organism_flag','id_confidence_known']
HIST=['comorbid_diabetes','comorbid_ckd_dialysis','comorbid_malignancy','comorbid_immunosuppression','comorbid_cardio_cerebrovascular','recent_hospitalization']
COURSE=['device_urinary_catheter','device_central_line','device_ventilator_tracheal','device_surgical_drain','icu_stay','prior_antibiotic_any','prior_antibiotic_named','recent_surgery']
NOTE=['note_available']
FS={'A':(NUM_A,BIN_A),
    'B_all':(NUM_A+['note_length_chars'],BIN_A+[c for c in cols if c!='note_length_chars']),
    'B_history_only':(NUM_A+['note_length_chars'],BIN_A+HIST+NOTE),
    'B_course_only':(NUM_A+['note_length_chars'],BIN_A+COURSE+NOTE),
    'B_recent_hosp_only':(NUM_A+['note_length_chars'],BIN_A+['recent_hospitalization']+NOTE),
    'B_note_metadata_only':(NUM_A+['note_length_chars'],BIN_A+NOTE),
    'B_history_flags_no_length':(NUM_A,BIN_A+HIST+NOTE)}
def prep(fs,mk):
    num,bin_=FS[fs]
    nump=Pipeline([('i',SimpleImputer(strategy='median',add_indicator=True))]+([('s',StandardScaler())] if mk=='logreg' else []))
    catp=Pipeline([('i',SimpleImputer(strategy='constant',fill_value='Missing')),('o',OneHotEncoder(handle_unknown='ignore'))])
    binp=Pipeline([('i',SimpleImputer(strategy='constant',fill_value=0))])
    return ColumnTransformer([('num',nump,num),('cat',catp,CAT_A),('bin',binp,bin_)],remainder='drop')
def model(mk,pw):
    if mk=='logreg': return LogisticRegression(max_iter=2000,class_weight='balanced',C=1.0,random_state=RS)
    if mk=='rf': return RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=5,class_weight='balanced',n_jobs=-1,random_state=RS)
    return xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,scale_pos_weight=pw,eval_metric='logloss',random_state=RS,n_jobs=-1)
def paired(y,pa,pb,pid,fn):
    rng=np.random.RandomState(123); d={}  # same seed/procedure as 10_delta_ci.py (Table 4)
    for i,p in enumerate(pid): d.setdefault(p,[]).append(i)
    keys=np.array(list(d.keys()),dtype=object); out=[]
    for _ in range(NB):
        idx=np.concatenate([d[k] for k in keys[rng.randint(0,len(keys),len(keys))]])
        if len(np.unique(y[idx]))<2: continue
        out.append(fn(y[idx],pb[idx])-fn(y[idx],pa[idx]))
    return float(np.mean(out)),float(np.percentile(out,2.5)),float(np.percentile(out,97.5))
rows=[]
TARGETS={'CRE':('in_CRE_target_cohort','label_CRE'),'ESBL':('in_ESBL_target_cohort','label_ESBL')}
for target,(flag,lab) in TARGETS.items():
  for subset in ['all']:  # note-present subset is reported by 17_sensitivity_analyses
    df=data[data[flag]&data[lab].notna()].copy(); df[lab]=df[lab].astype(int)
    if subset=='note_present': df=df[df.note_available==1]
    tr=df[df.split_temporal=='train']; te=df[df.split_temporal=='test']; y=te[lab].values; pid=te.study_id.values
    pw=(tr[lab]==0).sum()/max((tr[lab]==1).sum(),1)
    for mk in ['logreg','rf','xgb']:
        preds={}
        for fs in FS:
            p=Pipeline([('prep',prep(fs,mk)),('clf',model(mk,pw))]); p.fit(tr,tr[lab]); preds[fs]=p.predict_proba(te)[:,1]
        for fs in FS:
            if fs=='A': continue
            if subset=='note_present' and fs!='B_all': continue
            a=paired(y,preds['A'],preds[fs],pid,roc_auc_score); b=paired(y,preds['A'],preds[fs],pid,average_precision_score)
            rows.append(dict(target=target,subset=subset,model=mk,feature_set=fs,n_test=len(te),AUROC_A=round(roc_auc_score(y,preds['A']),4),AUROC_B=round(roc_auc_score(y,preds[fs]),4),
                 dAUROC=round(a[0],4),dAUROC_lo=round(a[1],4),dAUROC_hi=round(a[2],4),AUPRC_A=round(average_precision_score(y,preds['A']),4),AUPRC_B=round(average_precision_score(y,preds[fs]),4),
                 dAUPRC=round(b[0],4),dAUPRC_lo=round(b[1],4),dAUPRC_hi=round(b[2],4)))
            print(rows[-1], flush=True)
Path('tables/llm').mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv('tables/llm/table_llm_ablation_and_note_present.csv',index=False)
print('saved')
