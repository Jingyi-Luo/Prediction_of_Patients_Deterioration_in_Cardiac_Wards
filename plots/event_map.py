
# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

os.chdir('/Users/chloe/Desktop/UVa/Capstone/Code')
raw = pd.read_csv("../Data/moss_plos_one_data.csv")
true_data = pd.read_csv("../Data/true_data.csv")

id_true_list = list(set(true_data.id))

def event_map():
    patient_id = int(input("Patient's ID: "))
    y_start = -1
    test_patient = raw.loc[raw.id == patient_id,].reset_index()
    if patient_id in id_true_list:
        y_start = test_patient.loc[test_patient.y == True,].index[0]
    test_patient = test_patient.drop(['index','id', 'age', 'race', 'svc', 'LOS', 'ICU_Pt_Days', 'Mort', 'n_evts',
           'eid', 'y', 'tte', 'death', 'direct', 'MET', 'Sgy'],axis=1)
    test_patient = (test_patient - test_patient.mean()) / (test_patient.max()-test_patient.min())
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    hr = len(test_patient)/4
    binsx = (np.arange(0,hr,5)).astype(int)
    ax = sns.heatmap(test_patient.transpose().iloc[::-1],cmap="YlGnBu",xticklabels=20)
    title = "ID:" + str(patient_id) + " Outcome: {}".format("1" if patient_id in id_true_list else "0")
    ax.set_title(title)
    ax.set_xticklabels(binsx)
    plt.xlabel("Hour of stay")
    if y_start >= 0:
        plt.axvline(x=y_start,color='r',label="1")
    
event_map()
