import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import numpy as np

from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

#
working_dir = '/Users/ljyi/Desktop/capstone/capstone8'

os.chdir(working_dir)
#
raw_data = pd.read_csv('moss_plos_one_data.csv')
raw_data.columns = raw_data.columns.str.replace('.', '_')
raw_data.shape
# (2217958, 62)
col_names = raw_data.columns.tolist()

#==============================================================================
#                             Data Preprocessing
#==============================================================================
# find missing values
df = raw_data
df.head()
df_nan = df.isnull().sum(axis=0).to_frame()
df_nan.columns=['counts']
col_nan = df_nan[df_nan['counts']>0]
col_nan_index = list(col_nan.index)

# find unique values in 'id'
id_unique = df['id'].unique().tolist()
id_unique
len(id_unique)
# 8105

# get train and test index based on unique 'id'
import random
random.seed(1)
train_id = random.sample(id_unique, 5674)
test_id = [avar for avar in id_unique if avar not in train_id]

# get rid of variables with two many missing values
data_df = raw_data
drop_cols = ['n_evts', 'LOS', 'ICU_Pt_Days', 'Mort', 'age']  # why not age?
data_df.drop(col_nan_index, inplace=True, axis=1)
data_df.drop(drop_cols, inplace=True, axis=1)

# 'race' with three levels and 'svc' with four levels are categorical data
dummy_race = pd.get_dummies(data_df['race'])
data_df_dummy = pd.concat([data_df, dummy_race], axis=1)
data_df_dummy.drop(columns=['race', 'oth'], inplace=True, axis=1) # dummy variable trap

dummy_svc = pd.get_dummies(data_df['svc'])
df_svc_dummy = pd.concat([data_df_dummy, dummy_svc], axis=1)
df_svc_dummy.drop(columns=['svc', 'Other'], inplace=True, axis=1)

list(df_svc_dummy.columns)
df_dummy = df_svc_dummy

# split data into training and testing sets
df_dummy.set_index('id', inplace=True)
X_y_train = df_dummy.loc[train_id]
X_y_test = df_dummy.loc[test_id]

# sample training set
true_index = np.where(X_y_train['y'].values.flatten() == True)[0]
false_index = np.where(X_y_train['y'].values.flatten() == False)[0]
random.seed(0)
selected_false_index = random.sample(list(false_index), len(true_index)*2)
train_index = list(np.append(true_index, selected_false_index))
#
#true_index = np.where(X_y_test['y'].values.flatten() == True)[0]
#false_index = np.where(X_y_test['y'].values.flatten() == False)[0]
#random.seed(0)
#selected_false_index = random.sample(list(false_index), len(true_index)*2)
#test_index = list(np.append(true_index, selected_false_index))
# 
X_train = X_y_train.iloc[train_index, X_y_train.columns != 'y']
y_train = X_y_train.iloc[train_index, X_y_train.columns == 'y']
X_test = X_y_test.iloc[:, X_y_test.columns != 'y']
y_test = X_y_test.iloc[:, X_y_test.columns == 'y']
y_test = y_test.values.flatten()

len(y_train)
#1520840
np.sum(y_train == True)
# 16391
np.sum(y_train == False)
# 1504449
np.sum(y_test == True)
# 7490
np.sum(y_test == False)
# 689628

train_col_names = X_train.columns

# over-sampling using SMOTE-Synthetic Minority Oversampling Technique
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=train_col_names)
os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])

# check the lengths of data now
os_data_X.shape
# (2996702, 55)
len(os_data_y)
# 2996702
# percent of True
n_total = len(os_data_y)
n_true = sum(os_data_y['y']==True)
n_true
# 1498351 (before oversampling: 23881)

n_false = sum(os_data_y['y']==False)
n_false
# 1498351 (before oversampling:2194077)

pct_true = n_true/n_total
pct_true
# 0.5
# 50% are event
pct_false = n_false/n_total
pct_false
# 0.5
# 50% are non-event
# here, the ratio of event to non-event is 1:1 after SMOTE.

# Final data for training 
X_train_balanced = os_data_X
y_train_balanced = os_data_y

n_rows_total = len(y_train_balanced)
#n_rows_total_ls = range(n_rows_total)
random.seed(1)
#sample_rows_index = random.sample(n_rows_total_ls, 100000)
X_train_df = X_train_balanced
y_train_sample = y_train_balanced
y_train_sample = y_train_sample.values.flatten()

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sample = sc.fit_transform(X_train_df)  
X_test = sc.transform(X_test)
type(X_train_sample)

#==============================================================================
#                             Logistic Regression
#==============================================================================
from sklearn import metrics
logreg = LogisticRegression(random_state=0)
logreg.fit(X_train_sample, y_train_sample)

# Score 1 - choosing threshold
y_pred_proba_logreg = logreg.predict_proba(X_test)[:, 1]
threshold_S1_logreg = np.percentile(y_pred_proba_logreg,99)
threshold_S1_logreg # 0.893330285340538
y_pred_logreg_S1 = (y_pred_proba_logreg >= threshold_S1_logreg) 

# Confusion Matrix
cnf_matrix_logreg = metrics.confusion_matrix(y_test, y_pred_logreg_S1)
Se_logreg = cnf_matrix_logreg[1, 1]/(cnf_matrix_logreg[0, 1]+cnf_matrix_logreg[1, 1])
p_plus_logreg = cnf_matrix_logreg[1, 1]/(cnf_matrix_logreg[1, 1]+cnf_matrix_logreg[1, 0])
score_1_logreg = min(Se_logreg, p_plus_logreg)
print('Logistic Regression Score 1: {0:8.3f}'.format(score_1_logreg))
# 0.074

# ROC Curve: the receiver operating characteristic curve
from sklearn.metrics import roc_auc_score

y_pred_proba_logreg = logreg.predict_proba(X_test)[:, 1]
logit_roc_auc = roc_auc_score(y_test, y_pred_proba_logreg)
print('Logistic Regression roc: {0:8.3f}'.format(logit_roc_auc))
# 0.705 

# summary:
# Score 1: 0.021
# Score 1 with 1% threshold: 0.074
# AUC: 0.705

#==============================================================================
#                             Random Forest
#==============================================================================
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

forest_clf = RandomForestClassifier(n_estimators=600, n_jobs = -1, random_state=0)
forest_clf.fit(X_train_sample, y_train_sample)

# Score 1 - choosing threshold
y_pred_proba_RF = forest_clf.predict_proba(X_test)[:, 1]
threshold_S1_RF = np.percentile(y_pred_proba_RF,99)
threshold_S1_RF # 0.373
y_pred_RF_S1 = (y_pred_proba_RF >= threshold_S1_RF) 
# Confusion Matrix
cnf_matrix_RF = metrics.confusion_matrix(y_test, y_pred_RF_S1)
Se_RF = cnf_matrix_RF[1, 1]/(cnf_matrix_RF[0, 1]+cnf_matrix_RF[1, 1])
p_plus_RF = cnf_matrix_RF[1, 1]/(cnf_matrix_RF[1, 1]+cnf_matrix_RF[1, 0])
score_1_RF = min(Se_RF, p_plus_RF)
print('Logistic Regression Score 1: {0:8.3f}'.format(score_1_RF))
# 0.023

# ROC Curve: the receiver operating characteristic curve
y_pred_proba_forest = forest_clf.predict_proba(X_test)[:, 1]
forest_roc_auc = roc_auc_score(y_test, y_pred_proba_forest)
print('Random Forest roc: {0:8.3f}'.format(forest_roc_auc))
 
# summary:
# Score 1: 0.003
# Score 1 with 1% threshold: 0.024
# AUC: 0.709

