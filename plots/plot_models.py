import os
import matplotlib.pyplot as plt
import numpy as np  
#
working_dir = '/Users/ljyi/Desktop/capstone/capstone8'
os.chdir(working_dir)

# Bar plot for AUC and Score-1
#the commented code including ANN
#roc_list = [0.493, 0.705, 0.71 , 0.566, 0.631, 0.709]
#score_1_list = [0.686, 0.709, 0.732, 0.862, 0.946, 0.99 ]
#name_list = ['ANN', 'Logistic \nRegression', 'SVM\nLinear', 'KNN',
#       'SVM \nGaussian', 'Random \nForest']
roc_list = [0.705, 0.71 , 0.566, 0.631, 0.709]
score_1_list = [0.709, 0.732, 0.862, 0.946, 0.99 ]
name_list = ['Logistic \nRegression', 'SVM\nLinear', 'KNN',
       'SVM \nGaussian', 'Random \nForest']
x = np.arange(len(name_list))
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
fig = plt.figure(figsize=(8,4))
ax = fig.add_axes([0.10, 0.15, 0.85, 0.8])
ax.bar(x, height=score_1_list, width=0.4, color='black', alpha=0.8, label='Score 1')
ax.bar(x+0.4, height=roc_list, width=0.4, color='gray', alpha=0.8, label='ROC-AUC')
ax.legend()
ax.set_xticks(x+0.2)
ax.set_xticklabels(name_list)
ax.set_ylim(0.45, 1.0)
ax.grid(True)
ax.set_axisbelow(True)
plt.savefig('result_summary.png', dpi=300)


# ROC curves
myfontsize = 16
roc_random_forest = np.loadtxt('roc_random_forest.txt')
roc_logistic_regression = np.loadtxt('roc_logistic_regression.txt')
roc_svm_linear = np.loadtxt('roc_svm_linear.txt')
roc_svm_gaussian = np.loadtxt('roc_svm_nonlinear.txt')
roc_knn = np.loadtxt('roc_knn.txt')
#roc_ann = np.loadtxt('roc_ann.txt')
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.10, 0.10, 0.85, 0.85])
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

ax.plot(roc_random_forest[:,0], roc_random_forest[:,1], label = "Random Forest (area = {:0.2f})".format(roc_list[4]))
ax.plot(roc_logistic_regression[:,0], roc_logistic_regression[:,1], label = "Logistic Regression (area = {:0.2f})".format(roc_list[0]))
ax.plot(roc_svm_linear[:,0], roc_svm_linear[:,1], label = "SVM Linear (area = {:0.2f})".format(roc_list[1]))
ax.plot(roc_svm_gaussian[:,0], roc_svm_gaussian[:,1], label = "SVM Gaussian (area = {:0.2f})".format(roc_list[3]))
ax.plot(roc_knn[:,0], roc_knn[:,1], label = "KNN (area = {:0.2f})".format(roc_list[2]))
#ax.plot(roc_random_forest[:,0], roc_random_forest[:,1], label = "Random Forest (area = {:0.2f})".format(roc_list[5]))
#ax.plot(roc_logistic_regression[:,0], roc_logistic_regression[:,1], label = "Logistic Regression (area = {:0.2f})".format(roc_list[1]))
#ax.plot(roc_svm_linear[:,0], roc_svm_linear[:,1], label = "SVM Linear (area = {:0.2f})".format(roc_list[2]))
#ax.plot(roc_svm_gaussian[:,0], roc_svm_gaussian[:,1], label = "SVM Gaussian (area = {:0.2f})".format(roc_list[4]))
#ax.plot(roc_knn[:,0], roc_knn[:,1], label = "KNN (area = {:0.2f})".format(roc_list[3]))
#ax.plot(roc_ann[:,0], roc_ann[:,1], label = "ANN (area = {:0.2f})".format(roc_list[0]))
ax.plot([0, 1], [0, 1], '--', color='gray')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=myfontsize)
ax.set_ylabel('True Positive Rate', fontsize=myfontsize)
ax.set_title('Receiver Operatting Characteristic', fontsize=myfontsize)
ax.legend(loc='lower right')
plt.savefig('roc_summary.png', dpi=300)
