library(randomForest)
library(caret)
library(RANN)#for knn SMOTE stuff

generate_syn_data <- function(training){
  true_data_test <- data.frame(training$O2.Flow,training$edrk,training$WHITE.BLOOD.CELL.COUNT,training$hr)
  
  #scale the data so that k-nn isn't based off nominal differnces but distance from mean instead
  scaled.dat <- scale(true_data_test) 
  
  #finds 5 nearest neighbors for all points, first is always itself
  nearest <- nn2(scaled.dat,k = 6) 
  nearest <- as.data.frame(nearest)
  #data frame of nearest neighbor for each point in true data makes generating syntheic data easier
  nearest_neighbor <- true_data[nearest$nn.idx.2,]
  
  #synthtic data = (diff)*U(0,1) + old data where diff is difference between data and its nearest neighbor
  synthetic_data <- (nearest_neighbor[,16:61]-training[,16:61])*runif(length(16:61),min = 0, max =1)+training[,16:61]
  synthetic_data['y'] <- TRUE
  return(synthetic_data)
}

true_data <- readRDS(file ='C:\\Users\\x1\\Documents\\MSDS\\Capstone\\Data\\all_true.rds' )

#11993 is about 50/50 split with each containing seperate patients
train_true <- true_data[1:11993,]
test_true <- true_data[11994:23881,]

syn_train_true <- generate_syn_data(true_data[1:11993,])



load('C:\\Users\\x1\\Documents\\MSDS\\Capstone\\Data\\capstone_data_o2_filled.RData')
true_ids <- as.vector(unique(true_data[['id']]))


#generate null training set of similar size to event test size
train_null <- subset(all_data, id %in% 1:128)
saveRDS(train_null, file = "null_training_set.rds")
#make null test set
test_null <- subset(all_data, id %in% c(1050:1150,1181:1218))
saveRDS(train_null, file = "null_test_set.rds")

#takes up a bunch of space
rm(all_data)

#removes demographic columns 

train_true <- train_true[,c(10,16:61)]
train_null <- train_null[,c(10,16:61)]
test_true <- test_true[,c(10,16:61)]

#makes full training set w/ synthetic data
train <- rbind(train_null,train_true)
train <- rbind(train,syn_train_true)

#fits random forest to training set
test.rf=randomForest(as.factor(y) ~ .,data = train )
test.rf

# Call:
#   randomForest(formula = as.factor(y) ~ ., data = train) 
# Type of random forest: classification
# Number of trees: 500
# No. of variables tried at each split: 6
# 
# OOB estimate of  error rate: 0.4%
# Confusion matrix:
#   FALSE  TRUE  class.error
# FALSE 25986    17 0.0006537707
# TRUE    182 23804 0.0075877595


#finds variable importance of variables in random forest
import <- varImp(test.rf)
import
# Overall
# Glasgow.Coma.Scale.Total     193.44260
# O2.Flow                      521.23097
# Resp                        1091.65485
# SpO2                         355.05943
# SBP                          645.59628
# Pulse                       1681.45596
# Temp                         480.80169
# ALBUMIN                      316.83340
# ALKALINE.PHOSPHATASE         358.81521
# ALT.GPT                      375.34545
# AST.GOT                      523.61969
# BLOOD.UREA.NITROGEN         1114.59140
# CALCIUM                      797.70550
# CHLORIDE                     684.57696
# CO2                          968.02516
# CREATININE                   733.42884
# GLUCOSE                      542.57114
# HEMOGLOBIN                   751.96877
# LACTIC.ACID                  170.36895
# MAGNESIUM                    391.62602
# OXYGEN.SATURATION             94.15936
# PARTIAL.THROMBOPLASTIN.TIME  597.18492
# PCO2                         124.70517
# PHOSPHORUS                   730.39861
# PLATELET.COUNT               807.34705
# POTASSIUM                    641.98237
# PROTIME.INR                  640.33200
# SODIUM                       560.70049
# TOTAL.BILIRUBIN              873.91344
# TOTAL.PROTEIN                221.83611
# TROPONIN.I                  1590.47848
# WHITE.BLOOD.CELL.COUNT      1028.07621
# hr                          1405.84246
# s2.hr                        124.53267
# s8.hr                        232.13150
# s24.hr                       409.89814
# n.edrk                       143.14155
# edrk                         410.11846
# s2.edrk                       95.64147
# s8.edrk                      183.91072
# s24.edrk                     332.95666
# srr                          246.47668
# dfa                          161.84714
# cosen                        299.51172
# lds                          187.22365
# af                            68.01476

#checks mean of test events
#returns probability that event is true
predict(test.rf,test_true,type = "prob")[2] 
#.29

predict(test.rf,test_null,type = "prob")[2]
#.144


p0 <- predict(test.rf,test_null[sample(nrow(test_null),10000),],type = 'prob')[,2]
p1 <- predict(test.rf,test_true[sample(nrow(test_true),10000),],type = 'prob')[,2]
c_stat <- sum((p0 < p1))/10000
c_stat
#.6507 seems pretty reasonable
