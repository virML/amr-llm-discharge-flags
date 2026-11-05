"""Characteristics of temporal training and test sets (TRIPOD+AI items 16 and 20c). Output tables/supp/split_characteristics.csv"""
import pandas as pd, numpy as np
FL=['comorbid_diabetes','comorbid_ckd_dialysis','comorbid_malignancy','comorbid_immunosuppression','comorbid_cardio_cerebrovascular','prior_antibiotic_any','device_urinary_catheter','device_central_line','device_ventilator_tracheal','device_surgical_drain','icu_stay','recent_hospitalization','recent_surgery']
s=pd.read_parquet('data/processed/splits.parquet').reset_index(drop=True)
ll=pd.read_parquet('data/processed/nlp_features_llm.parquet').reset_index(drop=True)
rb=pd.read_parquet('data/processed/nlp_features.parquet').reset_index(drop=True)
assert (s[['study_id','isolate_id']].values==ll[['study_id','isolate_id']].values).all()
rows=[]
def add(name, fn): rows.append([name]+[fn(s[s.split_temporal==g].index) for g in ['train','test']])
pct=lambda m: f'{int(m.sum())} ({100*m.mean():.1f})'
add('Isolates, n', lambda i: str(len(i)))
add('Patients, n', lambda i: str(s.loc[i,'study_id'].nunique()))
add('Testing dates', lambda i: f"{s.loc[i,'testing_date'].min():%d %b %Y} to {s.loc[i,'testing_date'].max():%d %b %Y}")
add('Age, years, median (IQR)', lambda i: '{:.1f} ({:.1f} to {:.1f})'.format(*np.nanpercentile(s.loc[i,'age_years'].astype(float),[50,25,75])))
add('Age and gender not recorded, n (%)', lambda i: pct(s.loc[i,'age_years'].isna()))
# outpatient: no inpatient admission covering the testing date (ward unknown and admission-to-testing interval missing; see 24b)
add('Outpatient (not admitted at testing), n (%)', lambda i: pct((s.loc[i,'ward_clean']=='Unknown') & s.loc[i,'admission_to_culture_hours'].isna()))
add('Female, n (%)', lambda i: pct(s.loc[i,'gender_clean']=='Female'))
add('Paediatric, n (%)', lambda i: pct(s.loc[i,'is_pediatric_flag']==1))
for g,lab in [('Enterobacterales','Enterobacterales'),('Enterococcus','_Enterococcus_'),('Pseudomonas','_Pseudomonas_'),('Acinetobacter','_Acinetobacter_'),('Staph_aureus','_Staphylococcus aureus_')]:
    add(f'Organism group: {lab}, n (%)', lambda i,g=g: pct(s.loc[i,'organism_group']==g))
for sp in ['URINE','PUS','BLOOD']:
    add(f'Specimen: {sp.capitalize()}, n (%)', lambda i,sp=sp: pct(s.loc[i,'specimen_type_clean']==sp))
add('Admission-to-testing interval ≥48 h, n (%)', lambda i: pct(s.loc[i,'onset_bucket']=='Hospital-onset (>=48h)'))
add('Prior isolate from the same patient, n (%)', lambda i: pct(s.loc[i,'prior_isolate_count'].fillna(0)>0))
add('Note features available, n (%)', lambda i: pct(ll.loc[i,'note_available']==1))
add('Note length, characters, median (IQR)', lambda i: '{:.0f} ({:.0f} to {:.0f})'.format(*np.percentile(ll.loc[i][ll.loc[i,'note_available']==1]['note_length_chars'],[50,25,75])))
NM={'comorbid_diabetes':'Diabetes','comorbid_ckd_dialysis':'Chronic kidney disease or dialysis','comorbid_malignancy':'Malignancy','comorbid_immunosuppression':'Immunosuppression','comorbid_cardio_cerebrovascular':'Cardiovascular or cerebrovascular disease','prior_antibiotic_any':'Prior antibiotic exposure','device_urinary_catheter':'Urinary catheter','device_central_line':'Central line','device_ventilator_tracheal':'Mechanical ventilation or tracheal tube','device_surgical_drain':'Surgical drain','icu_stay':'ICU stay','recent_hospitalization':'Prior hospitalization','recent_surgery':'Recent surgery'}
rows.append(['__section__LLM-extracted flags among isolates with a note, n (%)','',''])
for f in FL:
    add(NM[f], lambda i,f=f: pct(ll.loc[i][ll.loc[i,'note_available']==1][f]==1))
rows.append(['__section__Resistance among labelled isolates, n/N (%)','',''])
for t,lab in [('MRSA','MRSA'),('ESBL','ESBL'),('CRE','CRE')]:
    add(lab, lambda i,t=t: (lambda x: f'{int((x==1).sum())}/{int(x.notna().sum())} ({100*(x==1).sum()/x.notna().sum():.1f})')(s.loc[i,f'label_{t}']))
out=pd.DataFrame(rows,columns=['characteristic','train','test']); out.to_csv('tables/supp/split_characteristics.csv',index=False); print(out.to_string(index=False))
