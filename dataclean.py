import csv
import pandas as pd

loc_path = 'mimic' 
data_1 = pd.read_csv(loc_path +'/Source.csv')
data_2 = pd.read_csv(loc_path +'/Target.csv')
data1 = data_1.copy()
data2 = data_2.copy()

data1['age'] = (data1['age']-data1['age'].mean())/data1['age'].std()
data1['triage_sbp'] = (data1['triage_sbp']-data1['triage_sbp'].mean())/data1['triage_sbp'].std()
data1['triage_dbp'] = (data1['triage_dbp']-data1['triage_dbp'].mean())/data1['triage_dbp'].std()
data1['triage_pain'] = (data1['triage_pain']-data1['triage_pain'].mean())/data1['triage_pain'].std()
data1['ed_heartrate_last'] = (data1['ed_heartrate_last']-data1['ed_heartrate_last'].mean())/data1['ed_heartrate_last'].std()
data1['ed_temperature_last'] = (data1['ed_temperature_last']-data1['ed_temperature_last'].mean())/data1['ed_temperature_last'].std()
data1['ed_resprate_last'] = (data1['ed_resprate_last']-data1['ed_resprate_last'].mean())/data1['ed_resprate_last'].std()
data1['ed_o2sat_last'] = (data1['ed_o2sat_last']-data1['ed_o2sat_last'].mean())/data1['ed_o2sat_last'].std()
data1['n_ed_90d'] = (data1['n_ed_90d']-data1['n_ed_90d'].mean())/data1['n_ed_90d'].std()

data2['age'] = (data2['age']-data1['age'].mean())/data1['age'].std()
data2['triage_sbp'] = (data2['triage_sbp']-data1['triage_sbp'].mean())/data1['triage_sbp'].std()
data2['triage_dbp'] = (data2['triage_dbp']-data1['triage_dbp'].mean())/data1['triage_dbp'].std()
data2['triage_pain'] = (data2['triage_pain']-data1['triage_pain'].mean())/data1['triage_pain'].std()
data2['ed_heartrate_last'] = (data2['ed_heartrate_last']-data1['ed_heartrate_last'].mean())/data1['ed_heartrate_last'].std()
data2['ed_temperature_last'] = (data2['ed_temperature_last']-data1['ed_temperature_last'].mean())/data1['ed_temperature_last'].std()
data2['ed_resprate_last'] = (data2['ed_resprate_last']-data1['ed_resprate_last'].mean())/data1['ed_resprate_last'].std()
data2['ed_o2sat_last'] = (data2['ed_o2sat_last']-data1['ed_o2sat_last'].mean())/data1['ed_o2sat_last'].std()
data2['n_ed_90d'] = (data2['n_ed_90d']-data1['n_ed_90d'].mean())/data1['n_ed_90d'].std()


data1.to_csv(loc_path+'/Source_c.csv')
data2.to_csv(loc_path+'/Target_c.csv')

