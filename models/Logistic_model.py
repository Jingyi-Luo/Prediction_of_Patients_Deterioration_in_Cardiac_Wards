import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import numpy as np
from sklearn.linear_model import LogisticRegression

#from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

#
working_dir = '/Users/ljyi/Desktop/capstone/capstone8/Logistic_regression'

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
drop_cols = ['n_evts', 'LOS', 'ICU_Pt_Days', 'Mort', 'age', 'race', 'svc']  # why not age?
data_df.drop(col_nan_index, inplace=True, axis=1)
data_df.drop(drop_cols, inplace=True, axis=1)

## 'race' with three levels and 'svc' with four levels are categorical data
#dummy_race = pd.get_dummies(data_df['race'])
#data_df_dummy = pd.concat([data_df, dummy_race], axis=1)
#data_df_dummy.drop(columns=['race', 'oth'], inplace=True, axis=1) # dummy variable trap
#
#dummy_svc = pd.get_dummies(data_df['svc'])
#df_svc_dummy = pd.concat([data_df_dummy, dummy_svc], axis=1)
#df_svc_dummy.drop(columns=['svc', 'Other'], inplace=True, axis=1)

list(data_df.columns)
df_dummy = data_df

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

# add shock index
#SI_train = X_train_df['Pulse']/X_train_df['SBP']
#SI_test = X_test['Pulse']/X_test['SBP']
#X_train_df['SI'] = SI_train
#X_test['SI'] = SI_test

## get column names of X_test
cols_test = X_test.columns

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sample = sc.fit_transform(X_train_df)  
X_test = sc.transform(X_test)
type(X_train_sample)

# change array to dataframe adding column names
X_train_sample = pd.DataFrame(X_train_sample)
X_train_sample.columns = X_train_df.columns
X_test = pd.DataFrame(X_test)
X_test.columns = cols_test


#==============================================================================
#                             Logistic Regression
#==============================================================================
from sklearn import metrics
from sklearn.metrics import roc_curve

logreg = LogisticRegression(random_state=0)
logreg.fit(X_train_sample, y_train_sample)

# Score 1 - choosing threshold
y_pred_proba_logreg = logreg.predict_proba(X_test)[:, 1]
threshold_S1_logreg = np.percentile(y_pred_proba_logreg,99)
threshold_S1_logreg # 0.893330285340538
y_pred_logreg_S1 = (y_pred_proba_logreg >= threshold_S1_logreg) 

# Confusion Matrix
cnf_matrix_logreg = metrics.confusion_matrix(y_test, y_pred_logreg_S1)
sensitivity_logreg = cnf_matrix_logreg[1, 1]/(cnf_matrix_logreg[0, 1]+cnf_matrix_logreg[1, 1]) # 
precision_logreg = cnf_matrix_logreg[1, 1]/(cnf_matrix_logreg[1, 1]+cnf_matrix_logreg[1, 0])
FPR_logreg = cnf_matrix_logreg[1, 0]/(cnf_matrix_logreg[0, 0]+cnf_matrix_logreg[1, 0]) # false positive rate for ROC curve
score_1_logreg = min(sensitivity_logreg, precision_logreg)
print('Logistic Regression Score 1: {0:8.3f}'.format(score_1_logreg))

# F1 score
F1_logreg = 2*sensitivity_logreg*precision_logreg/(sensitivity_logreg+precision_logreg)
print('Logistic Regression F1 Score: {0:8.3f}'.format(F1_logreg))

# ROC Curve: the receiver operating characteristic curve
from sklearn.metrics import roc_auc_score

y_pred_proba_logreg = logreg.predict_proba(X_test)[:, 1]
logit_roc_auc = roc_auc_score(y_test, y_pred_proba_logreg)
print('Logistic Regression roc: {0:8.3f}'.format(logit_roc_auc))

print('Logistic Sensivity (True Positive Rate): {0:8.3f}'.format(sensitivity_logreg))  
print('Logistic Precision: {0:8.3f}'.format(precision_logreg))
print('Logistic False Positive Rate: {0:8.3f}'.format(FPR_logreg))

# ======== with shock index (SI) =========
#Logistic Regression Score 1:    0.065
#Logistic Regression F1 Score:    0.065
#Logistic Regression roc:    0.715
#Logistic Sensivity (True Positive Rate):    0.066
#Logistic Precision:    0.065
#Logistic False Positive Rate:    0.010

# ======== without shock index (SI) =========
#Logistic Regression Score 1:    0.066
#Logistic Regression F1 Score:    0.067
#Logistic Regression roc:    0.710
#Logistic Sensivity (True Positive Rate):    0.068
#Logistic Precision:    0.066
#Logistic False Positive Rate:    0.010


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_logreg)
# 
fid = open('roc_logistic_regression.txt', 'w')
for i in range(len(fpr)):
    fid.write('{0:10.6f} {1:10.6f} {2:10.6f} \n'.format(fpr[i], tpr[i], thresholds[i]))
fid.close()
#
# logreg.classes_
# array([False,  True])

plt.figure(figsize=(8,6))
plt.rcParams['font.size'] = 18
plt.plot(fpr, tpr, label = "Logistic Regression (area = {:0.2f})".format(logit_roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operatting Characteristic: Logistic Regression')
plt.legend(loc='lower right')
plt.savefig('Log_ROC')
plt.close()


