"""Additional-file extras: full-corpus RB-vs-LLM agreement per flag, validation sample composition and
base-only precision, antibiogram by organism, structural missingness facts. Outputs tables/supp/."""
import pandas as pd, numpy as np, json, openpyxl
FL=['comorbid_diabetes','comorbid_ckd_dialysis','comorbid_malignancy','comorbid_immunosuppression','comorbid_cardio_cerebrovascular','prior_antibiotic_any','device_urinary_catheter','device_central_line','device_ventilator_tracheal','device_surgical_drain','icu_stay','recent_hospitalization','recent_surgery']
def kappa(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float); po=(y==p).mean(); pe=y.mean()*p.mean()+(1-y.mean())*(1-p.mean())
    return np.nan if pe==1 else (po-pe)/(1-pe)
# extraction-level analyses use all 12,930 extracted note instances (before 08b removed 182 notes written at a later admission from the modelling features)
import os
_sfx='_pre_datefix' if os.path.exists('data/processed/nlp_features_pre_datefix.parquet') else ''
rb=pd.read_parquet(f'data/processed/nlp_features{_sfx}.parquet').reset_index(drop=True)
ll=pd.read_parquet(f'data/processed/nlp_features_llm{_sfx}.parquet').reset_index(drop=True)
assert (rb[['study_id','isolate_id','testing_date']].values==ll[['study_id','isolate_id','testing_date']].values).all()  # positional alignment, as in modelling scripts
m=rb.note_available==1; assert (ll.note_available==1).equals(m); n=int(m.sum())
rows=[]; Y=[];P=[]
for f in FL:
    a=rb.loc[m,f].astype(int).values; b=ll.loc[m,f].astype(int).values; Y+=list(a); P+=list(b)
    rows.append(dict(flag=f,n_notes=n,rb_pos=int(a.sum()),rb_pct=100*a.mean(),llm_pos=int(b.sum()),llm_pct=100*b.mean(),
                     pct_agree=100*(a==b).mean(),kappa=kappa(a,b),both_pos=int(((a==1)&(b==1)).sum()),rb_only=int(((a==1)&(b==0)).sum()),llm_only=int(((a==0)&(b==1)).sum())))
Y=np.array(Y);P=np.array(P)
rows.append(dict(flag='POOLED',n_notes=n,rb_pos=int(Y.sum()),rb_pct=100*Y.mean(),llm_pos=int(P.sum()),llm_pct=100*P.mean(),pct_agree=100*(Y==P).mean(),kappa=kappa(Y,P),
                 both_pos=int(((Y==1)&(P==1)).sum()),rb_only=int(((Y==1)&(P==0)).sum()),llm_only=int(((Y==0)&(P==1)).sum())))
ag=pd.DataFrame(rows); ag.to_csv('tables/supp/corpus_agreement.csv',index=False); print(ag.round(3).to_string(index=False)); print('calls',len(Y))
# validation composition
KEY='data/deid/clinician_validation_sample_key_RESTRICTED.csv'  # written by 06; restricted (links review IDs to extraction outputs)
SHEET='paper/clinician_validation_review_sheet_completed.xlsx'  # workbook written by 06, completed by the physicians (final judgments)
key=pd.read_csv(KEY).set_index('review_id')
wb=openpyxl.load_workbook(SHEET,read_only=True,data_only=True)
rows_=list(wb['Review Sheet'].iter_rows(values_only=True)); h=rows_[0]; jc=[i for i,x in enumerate(h) if x and str(x).startswith('YOUR JUDGMENT')]
J=pd.DataFrame.from_dict({r[0]:[1 if (r[i] if i<len(r) else None)=='Present' else 0 for i in jc] for r in rows_[1:] if r and r[0]},orient='index',columns=FL)
base=key.index[key.sampling_stratum=='base_random']; enr=key[key.sampling_stratum=='enrichment']
print('enrichment_for counts:'); print(enr.enrichment_for.value_counts().to_string())
vr=[]
for meth in ['rb','llm']:
    for f in FL:
        own=enr.index[enr.enrichment_for.fillna('').apply(lambda x: f in str(x).split(','))]  # comma-separated membership, as in 06/07/19
        pb=base[key.loc[base,f'{meth}__{f}'].values==1]; tpb=int(J.loc[pb,f].sum())
        po=own[key.loc[own,f'{meth}__{f}'].values==1] if len(own) else own; tpo=int(J.loc[po,f].sum()) if len(po) else 0
        vr.append(dict(method=meth,flag=f,n_enrich=len(own),base_method_pos=len(pb),base_tp=tpb,base_only_precision=(tpb/len(pb) if len(pb) else np.nan),
                       enrich_method_pos=len(po),enrich_tp=tpo,ref_pos_base=int(J.loc[base,f].sum()),ref_pos_enrich=int(J.loc[own,f].sum()) if len(own) else 0))
vr=pd.DataFrame(vr); vr.to_csv('tables/supp/validation_composition.csv',index=False); print(vr.round(3).to_string(index=False))
# department composition of base sample
dcol=[c for c in key.columns if 'dept' in c.lower()]
print('key columns:',list(key.columns)[:12])
if dcol:
    d=key.loc[base,dcol[0]].value_counts(); print(d.to_string()); d.to_csv('tables/supp/base_sample_departments.csv')
# antibiogram
c=pd.read_parquet('data/processed/cohort_master.parquet')
ab=[]
for t,coh in [('MRSA','in_MRSA_target_cohort'),('ESBL','in_ESBL_target_cohort'),('CRE','in_CRE_target_cohort')]:
    s=c[c[coh]==True]
    g=s.groupby('organism_base').agg(n=('isolate_id','size'),n_lab=(f'label_{t}',lambda x:int(x.notna().sum())),n_pos=(f'label_{t}',lambda x:int((x==1).sum())))
    g=g[g.n_lab>0].sort_values('n_lab',ascending=False); g['pct']=100*g.n_pos/g.n_lab; g['target']=t
    tot=dict(organism_base='All',n=len(s),n_lab=int(s[f'label_{t}'].notna().sum()),n_pos=int((s[f'label_{t}']==1).sum())); tot['pct']=100*tot['n_pos']/tot['n_lab']; tot['target']=t
    ab+= g.reset_index().to_dict('records')+[tot]
ab=pd.DataFrame(ab); ab.to_csv('tables/supp/antibiogram.csv',index=False); print(ab.round(1).to_string(index=False))
# missingness facts (cleaned structured fields, as used in modelling and Table 1)
sf=pd.read_parquet('data/processed/structured_features.parquet').reset_index(drop=True)
unk=lambda x: x.astype(str).str.lower().isin(['unknown','missing','nan'])
miss=(sf['age_years'].isna()&unk(sf['gender_clean'])&unk(sf['ward_clean'])).values
fx=dict(n_block=int(miss.sum()),age_missing=int(sf['age_years'].isna().sum()),gender_unknown=int(unk(sf['gender_clean']).sum()),ward_unknown=int(unk(sf['ward_clean']).sum()),
        note_in_block=int(rb.loc[miss,'note_available'].sum()),note_overall=int((rb.note_available==1).sum()),
        collection_date_populated=int(c['Collection Date'].notna().sum()),id_conf_populated_final_cohort=int(c['ID Confidence'].notna().sum()))
print(fx); json.dump(fx,open('tables/supp/missingness_facts.json','w'),indent=1)
# within-run consistency of LLM extraction: discharge summaries linked to several isolates were extracted once per isolate
c2=pd.read_parquet('data/processed/cohort_master.parquet').reset_index(drop=True)
txt=c2['DISCHARGE_SUMMARY_CONTENT_DEID']; has=(ll.note_available==1).values
d=pd.DataFrame({'t':txt[has].values}); 
for f in FL: d[f]=ll.loc[has,f].values; d['rb_'+f]=rb.loc[has,f].values
sz=d.groupby('t').size(); dup=sz[sz>1]; dd=d[d.t.isin(dup.index)]
rows=[]; allc=[]
for f in FL:
    comp=[v for _,s in dd.groupby('t')[f] for v in (s.iloc[1:]==s.iloc[0]).tolist()]; allc+=comp
    rows.append(dict(flag=f,n_repeat_extractions=len(comp),pct_identical=100*np.mean(comp),docs_inconsistent=int((dd.groupby('t')[f].nunique()>1).sum())))
rows.append(dict(flag='POOLED',n_repeat_extractions=len(allc),pct_identical=100*np.mean(allc),docs_inconsistent=int((dd.groupby('t')[FL].nunique()>1).any(axis=1).sum())))
rc=pd.DataFrame(rows); rc.to_csv('tables/supp/llm_repeat_consistency.csv',index=False)
json.dump(dict(isolates_with_note=int(has.sum()),distinct_documents=int(txt[has].nunique()),docs_linked_to_multiple_isolates=int(len(dup)),isolates_in_those=int(dup.sum()),repeat_extractions=int(dup.sum()-len(dup))),open('tables/supp/document_counts.json','w'),indent=1)
print(rc.round(2).to_string(index=False)); print(json.load(open('tables/supp/document_counts.json')))
