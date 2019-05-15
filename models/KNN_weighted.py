import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import numpy as np

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
#                                   KNN
#==============================================================================

# ------------------------ Weighted KNN ---------------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score

# weight all features using random forest importance
feature_importance_forest = pd.read_csv('Feature Importance.csv',names = ['name','importance'])
X_train_knn = pd.DataFrame(X_train_sample, columns=X_train_df.columns)   # array to dataframe
X_test_knn = pd.DataFrame(X_test, columns=X_train_df.columns)
new_X_train = X_train_knn[sorted(X_train_knn.columns)]
new_X_test = X_test_knn[sorted(X_test_knn.columns)]
feature_importance_sorted = feature_importance_forest.sort_values('name')
array_X_train = np.array(new_X_train)      # 68412 * 52
array_X_test = np.array(new_X_test)
array_FI = np.array(feature_importance_sorted.iloc[:,1])   # 52 * 1
X_train_weighted = array_X_train * array_FI           # 68412 * 52
X_test_weighted = array_X_test * array_FI
X_train_weighted_df = pd.DataFrame(X_train_weighted, columns = sorted(X_train_df.columns))
X_test_weighted_df = pd.DataFrame(X_test_weighted, columns = sorted(X_train_df.columns))
#
knn_clf = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) #Euclidean
knn_clf.fit(X_train_weighted_df, y_train_sample)

# prediction X_test & accuracy score
y_pred_knn = knn_clf.predict(X_test_weighted_df)

# Score 1
cnf_matrix_knn = metrics.confusion_matrix(y_test, y_pred_knn)
Se_knn = cnf_matrix_knn[1, 1]/(cnf_matrix_knn[1, 1]+cnf_matrix_knn[1, 0])
p_plus_knn = cnf_matrix_knn[1, 1]/(cnf_matrix_knn[1, 1]+cnf_matrix_knn[0, 1])
score_1_knn = min(Se_knn, p_plus_knn)
print('KNN weighted Score 1: {0:8.3f}'.format(score_1_knn))
# 

# ROC Curve
y_pred_proba_knn = knn_clf.predict_proba(X_test)[:, 1]
knn_roc_auc = roc_auc_score(y_test, y_pred_proba_knn)
print('KNN weighted roc: {0:8.4f}'.format(knn_roc_auc))


# --------------------------- Non-weighted KNN --------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
#
knn_clf = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) #Euclidean
knn_clf.fit(X_train_sample, y_train_sample)

# prediction X_test & accuracy score
y_pred_knn = knn_clf.predict(X_test)
#metrics.accuracy_score(y_test, y_pred_knn)

# Score 1
cnf_matrix_knn = metrics.confusion_matrix(y_test, y_pred_knn)
Se_knn = cnf_matrix_knn[1, 1]/(cnf_matrix_knn[1, 1]+cnf_matrix_knn[1, 0])
p_plus_knn = cnf_matrix_knn[1, 1]/(cnf_matrix_knn[1, 1]+cnf_matrix_knn[0, 1])
score_1_knn = min(Se_knn, p_plus_knn)
print('KNN Score 1: {0:8.3f}'.format(score_1_knn))

# ROC Curve
y_pred_proba_knn = knn_clf.predict_proba(X_test)[:, 1]
knn_roc_auc = roc_auc_score(y_test, y_pred_proba_knn)
print('KNN roc: {0:8.4f}'.format(knn_roc_auc))
#

# Summary:
# Summary:
#KNN Score 1:    0.015
#KNN roc:   0.5659
#Weighted_KNN Score 1:    0.016
#Weighted_KNN roc:   0.4302      
