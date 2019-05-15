library(randomForest)
library(caret)
library(PRROC)
library(RANN)#for knn SMOTE stuff
#look into ranger for faster rf w/ weighting stuff
#makes synthetic sample with n*100% increase
generate_syn_data <- function(training,n){
  true_data_test <- data.frame(training$O2.Flow,training$edrk,training$WHITE.BLOOD.CELL.COUNT,training$hr)
  
  #scale the data so that k-nn isn't based off nominal differnces but distance from mean instead
  scaled.dat <- scale(true_data_test) 
  
  #finds 5 nearest neighbors for all points, first is always itself
  nearest <- nn2(scaled.dat,k = n+1) 
  nearest <- as.data.frame(nearest)
  #data frame of nearest neighbor for each point in true data makes generating syntheic data easier
  synthetic_data <- training[0,]
  for(i in 1:n){
  nearest_neighbor <- true_data[nearest[,i+1],]
  
  #synthtic data = (diff)*U(0,1) + old data where diff is difference between data and its nearest neighbor
  current <- (nearest_neighbor[,16:61]-training[,16:61])*runif(length(16:61),min = 0, max =1)+training[,16:61]
  current['y'] <- TRUE
  synthetic_data <- merge(synthetic_data,current,all = TRUE)
  }
  
  return(synthetic_data)
}

true_data <- readRDS(file ='C:\\Users\\x1\\Documents\\MSDS\\Capstone\\Data\\all_true.rds' )
load('C:\\Users\\x1\\Documents\\MSDS\\Capstone\\Data\\capstone_data_o2_filled.RData')


true_ids <- as.vector(unique(true_data[['id']]))
null_ids <- setdiff(as.vector(unique(all_data[['id']])),true_ids)

t1 <- true_ids[1:74]
t2 <- true_ids[75:148]
t3 <- true_ids[149:222]
t4 <- true_ids[223:295]
t5 <- true_ids[296:367]

count = 1
c_stat = rep(0,5)
precision_recall <- rep(0,5)
mean_null <-  rep(0,5)
mean_true <- rep(0,5)
for(i in 1:5){#in c(t1,t2,t3,t4,t5)
  if(i == 1){
    ids = t1
  }
  if(i == 2){
    ids = t2
  }
  if(i == 3){
    ids = t3
  }
  if(i == 4){
    ids = t4
  }
  if(i == 5){
    ids = t5
  }
  #remove testing set from training
  train <- subset(true_data,!(id %in% ids))
 
  #add synthetic samples
  train <- rbind(train,generate_syn_data(train,2))#should be about 38210
  #average case has 270 observations so roughly match events to non-events by changing
  #second number 
  null_training <- sample(null_ids,140)#70 for no SMOTE
  
  #add null samples to training total data set
  train <- rbind(train,subset(all_data,id %in% null_training))
  #remove demographic information
  train <- train[,c(10,16:61)]
  
  #fit random forest model
  test.rf=randomForest(as.factor(y) ~ .,data = train,ntree=100)
  
  #create test set
  test_true <- subset(true_data,(id %in% ids))
  #predict probs for test set
  p1 <- predict(test.rf,test_true,type = 'prob')[,2]
  #generate null-test set
  test_null <- subset(all_data,!((id %in% null_training) | ((id %in% ids)& y == FALSE)))
  p0 <- predict(test.rf,test_null[sample(nrow(test_null),50000),],type = 'prob')[,2]
  rm(test_null)
  mean_true[i] <- mean(p1)
  mean_null[i] <- mean(p0)
  c_stat[count] <- roc.curve(scores.class0 = p1, scores.class1 = p0)$auc
  precision_recall[count] <- pr.curve(scores.class0 = p1, scores.class1 = p0)$auc.integral
  
  print(count)
  count = count + 1
}

##################
#SMOTE Tests
##################
#tests all had same null sample and event samples
mean(c_stat)
#0.6556 0.6249 0.6692 0.7137 0.6555 No Smote same null-sample
#mean: 0.66378
#mean: 0.70196 100% SMOTE 
#mean: 0.69876 200% SMOTE

mean(mean_true)
#0.5667756 No SMOTE
#0.7408276 100%
#0.7687924 200%

mean(mean_null)
#0.433344 No SMOTE
#0.5882268 100%
#0.6267252 200%
# 300% with 5/1 sampling with 10/1 weigths
# > mean(c_stat)
# [1] 0.74346
# > mean(mean_true)
# [1] 0.456225
# > mean(mean_null)
# [1] 0.2234312
#below tests were with 50/50 split event to non-event

c_stat
#0.6345 0.6138 0.6115 0.6791 0.6003 no SMOTE

#0.7407 0.7007 0.6436 0.7970 0.6134 100% SMOTE 50/50 event non-event

#0.7588 0.7627 0.6887 0.7912 0.6212 200% SMOTE 200%

mean_true
#0.6763492 0.7091872 0.7268920 0.6877532 0.7328020 No SMOTE

#0.7585836 0.7547308 0.7216926 0.7287996 0.5687726 200%

mean_null
#0.5919854 0.6379786 0.6464654 0.5791818 0.6693016 No SMOTE

#0.5640024 0.5529552 0.5823842 0.5161234 0.4767196 200%



####################
#N-Trees Experiment
####################

#50 Trees
# c_stats
#0.6269 0.5950 0.5883 0.6771 0.6381
#mean c_stat: .62508

mean_true
#0.682866 0.665852 0.656508 0.732516 0.676786
mean(mean_true)
#0.6829056
#mean_null
#0.588838 0.588364 0.587366 0.602988 0.566024
mean(mean_null)
#0.586716

#100 Trees
c_stat
#0.5915 0.6150 0.6549 0.6999 0.5807
#mean c_stat: 0.6284

mean_true
#0.687082 0.705597 0.688049 0.745624 0.685714
mean(mean_true)
#0.7024132

mean_null
#0.634697 0.624996 0.590445 0.608334 0.628196
mean(mean_null)
#0.6173336

#500 Trees
c_stat
#0.6345 0.6138 0.6115 0.6791 0.6003
#mean:0.62784
mean_null
#0.5919854 0.6379786 0.6464654 0.5791818 0.6693016

##################
#Class weights 
##################
#c(2,1) w/ roughly 2:1 non-event to event
c_stat
#0.6542 0.6291 0.6547 0.6605 0.6230
mean_true
#0.560414 0.545017 0.564984 0.611126 0.564297
mean_null
#0.437819 0.449940 0.446664 0.478857 0.461206

#c(1,2) w/ roughly 3:1 non-event to event
c_stat
#0.6263 0.6461 0.6518 0.6909 0.6171
mean_true
#0.423919 0.390815 0.444191 0.399637 0.395341
mean_null
#0.338176 0.308476 0.342510 0.302007 0.319527

#c(1,10) w/ roughly 3:1 non-event to event
c_stat
#0.6123 0.5845 0.6161 0.6004 0.6435
mean_true
#0.355080 0.320811 0.336549 0.352563 0.422466
mean_null
#0.281983 0.283129 0.276222 0.292102 0.331318

#c(10,1) w/ roughly 3:1 non-event to event
c_stat
#0.6810 0.6216 0.6629 0.7355 0.6469
mean_true
#0.505786 0.466883 0.550536 0.610326 0.528074
mean_null
#0.356917 0.366829 0.407562 0.393811 0.381064