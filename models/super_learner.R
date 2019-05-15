library(randomForest)
library(caret)
library(xgboost)
library(MLmetrics)
library(e1071)
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
    current <- (nearest_neighbor[,c(16:61)]-training[,c(16:61)])*runif(length(16:61),min = 0, max =1)+training[,c(16:61)]
    current['y'] <- TRUE
    #current['PT_sex'] <- training$PT_sex
    synthetic_data <- merge(synthetic_data,current,all = TRUE)
  }
  
  return(synthetic_data)
}
predict_labels <- function(test,model,model_type,num_1){
  if(model_type == 'rf'){
  probs <- predict(model,test,type = "prob")[,2]
  }
  if(model_type == 'xg'){
    probs <- predict(model,as.matrix(test[,-c(1,2)]))
  }
  if(model_type == 'lg'){
    probs <- predict(model,test, type = "response")
  }
  if(model_type == 'nb'){
    probs = predict(model,test,"raw")[,2]
  }
  index <- order(probs, decreasing=TRUE)[1:num_1][num_1]
  preds <- rep(0,length(test$y))
  preds[which(probs >= probs[index])] <- 1
  output <- list("probs" = probs,"preds" = preds)
  return(output)
  
}
#Read in Data
load('C:\\Users\\x1\\Documents\\MSDS\\Capstone\\Data\\capstone_data_o2_filled.RData')

all_data <- all_data[seq(1,nrow(all_data),4),]
#Seperate True Values
true_data <- all_data[which(all_data$y == TRUE),]

true_ids <- as.vector(unique(true_data[['id']]))
null_ids <- setdiff(as.vector(unique(all_data[['id']])),true_ids)



#Setup Cross Validation Data Sets
t1 <- true_ids[1:72]#73,74
t2 <- true_ids[75:146]#147,148
t3 <- true_ids[149:220]#221,222
t4 <- true_ids[223:293]#294.295
t5 <- true_ids[296:365]#367.366
set.seed(187)
nt1 <- sample(null_ids,1538)
nt2 <- sample(setdiff(null_ids,nt1),1538)
nt3 <- sample(setdiff(null_ids,c(nt1,nt2)),1538)
nt4 <- sample(setdiff(null_ids,c(nt1,nt2,nt3)),1538)
nt5 <- setdiff(null_ids,c(nt1,nt2,nt3,nt4))

#Initialize all needed lists
count = 1
rf_preds.labs = c()
lg_preds.labs = c()
xg_preds.labs = c()
rf_probs.labs = c()
lg_probs.labs = c()
xg_probs.labs = c()
rf_preds.labs.vitals = c()
lg_preds.labs.vitals = c()
xg_preds.labs.vitals = c()
rf_probs.labs.vitals = c()
lg_probs.labs.vitals = c()
xg_probs.labs.vitals = c()
rf_preds.labs.ecg = c()
lg_preds.labs.ecg = c()
xg_preds.labs.ecg = c()
rf_probs.labs.ecg = c()
lg_probs.labs.ecg = c()
xg_probs.labs.ecg = c()
rf_preds.vitals = c()
lg_preds.vitals = c()
xg_preds.vitals = c()
rf_probs.vitals = c()
lg_probs.vitals = c()
xg_probs.vitals = c()
rf_preds.vitals.ecg = c()
lg_preds.vitals.ecg = c()
xg_preds.vitals.ecg = c()
rf_probs.vitals.ecg = c()
lg_probs.vitals.ecg = c()
xg_probs.vitals.ecg = c()
rf_preds.ecg = c()
lg_preds.ecg = c()
xg_preds.ecg = c()
rf_probs.ecg = c()
lg_probs.ecg = c()
xg_probs.ecg = c()
rf_preds.all = c()
lg_preds.all = c()
xg_preds.all = c()
rf_probs.all = c()
lg_probs.all = c()
xg_probs.all = c()
nb.preds.labs = c()
nb.preds.vitals = c()
nb.preds.ecg = c()
nb.preds.labs.vitals = c()
nb.preds.vitals.ecg = c()
nb.preds.labs.ecg = c()
nb.probs.labs = c()
nb.probs.vitals = c()
nb.probs.labs.vitals = c()
nb.probs.vitals.ecg = c()
nb.probs.labs.ecg = c()
nb.probs.ecg = c()

test_y <- c()
test_ids <- c()

labs = c(10:34)
vitals <- c(3:9)
ecg <- c(35:48)

for(i in 1:5){#in c(t1,t2,t3,t4,t5)
  #split into training and test sets
  if(i == 1){
    ids = t1
    train_true <- c(t2,t3,t4,t5)
    nulls = c(nt2,nt3,nt4,nt5)
    test_null <- nt1
  }
  if(i == 2){
    ids = t2
    nulls = c(nt1,nt3,nt4,nt5)
    train_true <- c(t1,t3,t4,t5)
    test_null <- nt2
  }
  if(i == 3){
    ids = t3
    train_true <- c(t2,t1,t4,t5)
    nulls = c(nt1,nt2,nt4,nt5)
    test_null <- nt3
  }
  if(i == 4){
    ids = t4
    train_true <- c(t2,t3,t1,t5)
    nulls = c(nt1,nt3,nt2,nt5)
    test_null <- nt4
  }
  if(i == 5){
    ids = t5
    train_true <- c(t2,t3,t4,t1)
    nulls = c(nt1,nt3,nt4,nt2)
    test_null <- nt5
  }
  #remove testing set from training
  train <- subset(true_data,!(id %in% ids))
  
  #add synthetic samples
  train <- rbind(train,generate_syn_data(train,4))#should be about 38210

  train <- rbind(train,subset(all_data,id %in% train_true & y == 0)[seq(1,nrow(all_data[which(all_data$id %in% train_true & all_data$y == 0),]),6),])
  
  null_training <- nulls#70 for no SMOTE
  
  #add null samples to training total data set
  train <- rbind(train,subset(all_data,id %in% null_training)[seq(1,nrow(all_data[which(all_data$id %in% null_training),]),6),])
  #remove demographic information
  train <- train[,c(10,1,16:61)]#,63:64)]
  
  
  #fit random forest model
  print('RF Model')
  rf.mdl.labs =randomForest(as.factor(y) ~ .,data = train[,c(1,labs)],ntree=100,classwt = c(1,.2))
  rf.mdl.labs.vitals =randomForest(as.factor(y) ~ .,data = train[,c(1,labs,vitals)],ntree=100,classwt = c(1,.2))
  rf.mdl.labs.ecg =randomForest(as.factor(y) ~ .,data = train[,c(1,labs,ecg)],ntree=100,classwt = c(1,.2))
  rf.mdl.vitals =randomForest(as.factor(y) ~ .,data = train[,c(1,vitals)],ntree=100,classwt = c(1,.2))
  rf.mdl.vitals.ecg =randomForest(as.factor(y) ~ .,data = train[,c(1,vitals,ecg)],ntree=100,classwt = c(1,.2))
  rf.mdl.all =randomForest(as.factor(y) ~ .,data = train[,c(1,vitals,labs,ecg)],ntree=100,classwt = c(1,.2))
  rf.mdl.ecg =randomForest(as.factor(y) ~ .,data = train[,c(1,ecg)],ntree=100,classwt = c(1,.2))

  print('Logisitc Regression running')
  lg.mod.labs <- glm(as.factor(y) ~.,data = train[,c(1,labs)],family = 'binomial')
  lg.mod.labs.ecg <- glm(as.factor(y) ~.,data = train[,c(1,labs,ecg)],family = 'binomial')
  lg.mod.labs.vitals <- glm(as.factor(y) ~.,data = train[,c(1,labs,vitals)],family = 'binomial')
  lg.mod.vitals.ecg <- glm(as.factor(y) ~.,data = train[,c(1,vitals,ecg)],family = 'binomial')
  lg.mod.all <- glm(as.factor(y) ~.,data = train[,c(1,vitals,labs,ecg)],family = 'binomial')
  lg.mod.vitals <- glm(as.factor(y) ~.,data = train[,c(1,vitals)],family = 'binomial')
  lg.mod.ecg <- glm(as.factor(y) ~.,data = train[,c(1,ecg)],family = 'binomial')
  
  print("XG Boost")
  xg.mod.labs <- xgboost(data = as.matrix(train[,labs]), label = train$y, max.depth = 3, eta = 1, nthread = 2, nrounds = 8, objective = "binary:logistic")
  xg.mod.labs.ecg <- xgboost(data = as.matrix(train[,c(labs,ecg)]), label = train$y, max.depth = 3, eta = 1, nthread = 2, nrounds = 8, objective = "binary:logistic")
  xg.mod.labs.vitals <- xgboost(data = as.matrix(train[,c(labs,vitals)]), label = train$y, max.depth = 3, eta = 1, nthread = 2, nrounds = 8, objective = "binary:logistic")
  xg.mod.vitals <- xgboost(data = as.matrix(train[,vitals]), label = train$y, max.depth = 3, eta = 1, nthread = 2, nrounds = 8, objective = "binary:logistic")
  xg.mod.vitals.ecg <- xgboost(data = as.matrix(train[,c(vitals,ecg)]), label = train$y, max.depth = 3, eta = 1, nthread = 2, nrounds = 8, objective = "binary:logistic")
  xg.mod.all <- xgboost(data = as.matrix(train[,c(vitals,labs,ecg)]), label = train$y, max.depth = 3, eta = 1, nthread = 2, nrounds = 8, objective = "binary:logistic")
  xg.mod.ecg <- xgboost(data = as.matrix(train[,ecg]), label = train$y, max.depth = 3, eta = 1, nthread = 2, nrounds = 8, objective = "binary:logistic")

  print("Naive Bayes")
  nb.mod.labs <- naiveBayes(as.factor(y) ~ .,data = train[,c(1,labs)])
  nb.mod.labs.vitals <- naiveBayes(as.factor(y) ~ .,data = train[,c(1,labs,vitals)])
  nb.mod.labs.ecg <- naiveBayes(as.factor(y) ~ .,data = train[,c(1,labs,ecg)])
  nb.mod.vitals <- naiveBayes(as.factor(y) ~ .,data = train[,c(1,vitals)])
  nb.mod.vitals.ecg <- naiveBayes(as.factor(y) ~ .,data = train[,c(1,vitals,ecg)])
  nb.mod.ecg <- naiveBayes(as.factor(y) ~ .,data = train[,c(1,ecg)])

  # if(i ==1){
  #   import.labs <- varImp(rf.mdl.labs)
  #   import.ecg <- varImp(rf.mdl.ecg)
  #   import.vitals <- varImp(rf.mdl.vitals)
  # }
  # else{
  #   import.labs <- cbind(import.labs,varImp(rf.mdl.labs))
  #   import.ecg <- cbind(import.ecg,varImp(rf.mdl.ecg))
  #   import.vitals <- cbind(import.vitals,varImp(rf.mdl.vitals))
  # }
  
  test <- subset(all_data,id %in% ids)
  num_1 <- length(test$y[test$y == 1])
  test <- rbind(test,subset(all_data,id %in% test_null))
  #null_needed <- 99*num_1 - length(test$y) 
  #test <- rbind(test,subset(all_data,!(id %in% nulls) & !(id %in% true_ids))[sample(length(subset(all_data,!(id %in% nulls) & !(id %in% true_ids))$y),null_needed),])
  test <- test[,c(1,10,16:61)]
 
  print("Making predections")
  
  output <- predict_labels(test,rf.mdl.labs.vitals,'rf',num_1)
  rf_preds.labs.vitals <- c(rf_preds.labs.vitals,output$preds)
  rf_probs.labs.vitals <- c(rf_probs.labs.vitals,unname(output$probs))

  output <- predict_labels(test[,c(1,2,labs,vitals)],xg.mod.labs.vitals,'xg',num_1)
  xg_preds.labs.vitals <- c(xg_preds.labs.vitals,output$preds)
  xg_probs.labs.vitals <- c(xg_probs.labs.vitals,output$probs)

  output <- predict_labels(test,lg.mod.labs.vitals,'lg',num_1)
  lg_preds.labs.vitals <- c(lg_preds.labs.vitals,output$preds)
  lg_probs.labs.vitals <- c(lg_probs.labs.vitals,unname(output$probs))

  output <- predict_labels(test,rf.mdl.labs.ecg,'rf',num_1)
  rf_preds.labs.ecg <- c(rf_preds.labs.ecg,output$preds)
  rf_probs.labs.ecg <- c(rf_probs.labs.ecg,unname(output$probs))

  output <- predict_labels(test[,c(1,2,labs,ecg)],xg.mod.labs.ecg,'xg',num_1)
  xg_preds.labs.ecg <- c(xg_preds.labs.ecg,output$preds)
  xg_probs.labs.ecg <- c(xg_probs.labs.ecg,output$probs)
  
  output <- predict_labels(test,lg.mod.labs.ecg,'lg',num_1)
  lg_preds.labs.ecg <- c(lg_preds.labs.ecg,output$preds)
  lg_probs.labs.ecg <- c(lg_probs.labs.ecg,unname(output$probs))

  output <- predict_labels(test,rf.mdl.labs,'rf',num_1)
  rf_preds.labs <- c(rf_preds.labs,output$preds)
  rf_probs.labs <- c(rf_probs.labs,unname(output$probs))

  output <- predict_labels(test[,c(1,2,labs)],xg.mod.labs,'xg',num_1)
  xg_preds.labs <- c(xg_preds.labs,output$preds)
  xg_probs.labs <- c(xg_probs.labs,output$probs)
  
  output <- predict_labels(test,lg.mod.labs,'lg',num_1)
  lg_preds.labs <- c(lg_preds.labs,output$preds)
  lg_probs.labs <- c(lg_probs.labs,unname(output$probs))

  output <- predict_labels(test,nb.mod.labs,'nb',num_1)
  nb.preds.labs <- c(nb.preds.labs,output$preds)
  nb.probs.labs <- c(nb.probs.labs,unname(output$probs))

  output <- predict_labels(test,nb.mod.labs.vitals,'nb',num_1)
  nb.preds.labs.vitals <- c(nb.preds.labs.vitals,output$preds)
  nb.probs.labs.vitals <- c(nb.probs.labs.vitals,unname(output$probs))

  output <- predict_labels(test,nb.mod.labs.ecg,'nb',num_1)
  nb.preds.labs.ecg <- c(nb.preds.labs.ecg,output$preds)
  nb.probs.labs.ecg <- c(nb.probs.labs.ecg,unname(output$probs))

  output <- predict_labels(test,nb.mod.vitals,'nb',num_1)
  nb.preds.vitals <- c(nb.preds.vitals,output$preds)
  nb.probs.vitals <- c(nb.probs.vitals,unname(output$probs))

  output <- predict_labels(test,nb.mod.vitals.ecg,'nb',num_1)
  nb.preds.vitals.ecg <- c(nb.preds.vitals.ecg,output$preds)
  nb.probs.vitals.ecg <- c(nb.probs.vitals.ecg,unname(output$probs))

  output <- predict_labels(test,nb.mod.ecg,'nb',num_1)
  nb.preds.ecg <- c(nb.preds.ecg,output$preds)
  nb.probs.ecg <- c(nb.probs.ecg,unname(output$probs))

  output <- predict_labels(test,rf.mdl.vitals,'rf',num_1)
  rf_preds.vitals <- c(rf_preds.vitals,output$preds)
  rf_probs.vitals <- c(rf_probs.vitals,unname(output$probs))

  output <- predict_labels(test[,c(1,2,vitals)],xg.mod.vitals,'xg',num_1)
  xg_preds.vitals <- c(xg_preds.vitals,output$preds)
  xg_probs.vitals <- c(xg_probs.vitals,output$probs)

  output <- predict_labels(test,lg.mod.vitals,'lg',num_1)
  lg_preds.vitals <- c(lg_preds.vitals,output$preds)
  lg_probs.vitals <- c(lg_probs.vitals,unname(output$probs))

  output <- predict_labels(test,rf.mdl.vitals.ecg,'rf',num_1)
  rf_preds.vitals.ecg <- c(rf_preds.vitals.ecg,output$preds)
  rf_probs.vitals.ecg <- c(rf_probs.vitals.ecg,unname(output$probs))

  output <- predict_labels(test[,c(1,2,vitals,ecg)],xg.mod.vitals.ecg,'xg',num_1)
  xg_preds.vitals.ecg <- c(xg_preds.vitals.ecg,output$preds)
  xg_probs.vitals.ecg <- c(xg_probs.vitals.ecg,output$probs)

  output <- predict_labels(test,lg.mod.vitals.ecg,'lg',num_1)
  lg_preds.vitals.ecg <- c(lg_preds.vitals.ecg,output$preds)
  lg_probs.vitals.ecg <- c(lg_probs.vitals.ecg,unname(output$probs))

  output <- predict_labels(test,rf.mdl.ecg,'rf',num_1)
  rf_preds.ecg <- c(rf_preds.ecg,output$preds)
  rf_probs.ecg <- c(rf_probs.ecg,unname(output$probs))

  output <- predict_labels(test[,c(1,2,ecg)],xg.mod.ecg,'xg',num_1)
  xg_preds.ecg <- c(xg_preds.ecg,output$preds)
  xg_probs.ecg <- c(xg_probs.ecg,output$probs)

  output <- predict_labels(test,lg.mod.ecg,'lg',num_1)
  lg_preds.ecg <- c(lg_preds.ecg,output$preds)
  lg_probs.ecg <- c(lg_probs.ecg,unname(output$probs))

  output <- predict_labels(test,rf.mdl.all,'rf',num_1)
  rf_preds.all <- c(rf_preds.all,output$preds)
  rf_probs.all <- c(rf_probs.all,unname(output$probs))

  output <- predict_labels(test[,c(1,2,vitals,labs,ecg)],xg.mod.all,'xg',num_1)
  xg_preds.all <- c(xg_preds.all,output$preds)
  xg_probs.all <- c(xg_probs.all,output$probs)

  output <- predict_labels(test,lg.mod.all,'lg',num_1)
  lg_preds.all <- c(lg_preds.all,output$preds)
  lg_probs.all <- c(lg_probs.all,unname(output$probs))

  
  test_y <- c(test_y,test$y)
  test_ids <- c(test_ids,test$id)
  
  print(count)
  count = count + 1
}
results <- data.frame(ids = test_ids,y = test_y,rf_preds.labs = rf_preds.labs,rf_preds.vitals = rf_preds.vitals,rf_preds.ecg = rf_preds.ecg,lg_preds.labs = lg_preds.labs,lg_preds.vitals = lg_preds.vitals,lg_preds.ecg = lg_preds.ecg,xg_preds.labs = xg_preds.labs,xg_preds.vitals = xg_preds.vitals,xg_preds.ecg = xg_preds.ecg)
results$y[which(results$y == 'TRUE')] <- 1
results$y[which(results$y == 'FALSE')] <- 0
# 
F1_Score(y_pred = rf_preds.labs.ecg, y_true = results$y, positive = 1)
#0.172 now .22
# F1_Score(y_pred = results$rf_preds.ecg, y_true = results$y, positive = 1)
# #0.055
# F1_Score(y_pred = results$rf_preds.vitals, y_true = results$y, positive = 1)
# #.0726
# F1_Score(y_pred = results$lg_preds.vitals, y_true = results$y, positive = 1)
# #0.08
# F1_Score(y_pred = results$lg_preds.labs, y_true = results$y, positive = 1)
# #0.033
# F1_Score(y_pred = results$lg_preds.ecg, y_true = results$y, positive = 1)
# #.04
# F1_Score(y_pred = results$xg_preds.ecg, y_true = results$y, positive = 1)
# #0.045
# F1_Score(y_pred = results$xg_preds.labs, y_true = results$y, positive = 1)
# #0.052
# F1_Score(y_pred = results$xg_preds.vitals, y_true = results$y, positive = 1)
# #0.057
library(pROC)
auc(results$y,rf_probs.labs)

# labs_found <- unique(results$ids[which((results$rf_preds.labs == 1 | results$lg_preds.labs == 1 | results$xg_preds.labs == 1) & results$y )])
# ecg_found <- unique(results$ids[which((results$rf_preds.ecg == 1 | results$lg_preds.ecg == 1 | results$xg_preds.ecg == 1) & results$y )])
# vitals_found <- unique(results$ids[which((results$rf_preds.vitals == 1 | results$lg_preds.vitals == 1 | results$xg_preds.vitals == 1) & results$y )])
# labs_only <- setdiff(labs_found,c(ecg_found,vitals_found))
# vitals_only <- setdiff(vitals_found,c(ecg_found,labs_found))
# ecg_only <- setdiff(ecg_found,c(vitals_found,labs_found))
# all_found <- unique(c(labs_found,ecg_found,vitals_found))
# 
# rf_labs_found <- unique(results$ids[which(results$rf_preds.labs == 1 & results$y)])
# lg_labs_found <- unique(results$ids[which(results$lg_preds.labs == 1 & results$y)])
# xg_labs_found <- unique(results$ids[which(results$xg_preds.labs == 1 & results$y )])
# rf_labs_only <- setdiff(rf_labs_found,c(lg_labs_found,xg_labs_found))
# xg_labs_only <- setdiff(xg_labs_found,c(lg_labs_found,rf_labs_found))
# 
# 
# rf_vitals_found <- unique(results$ids[which(results$rf_preds.vitals == 1 & results$y)])
# lg_vitals_found <- unique(results$ids[which(results$lg_preds.vitals == 1 & results$y)])
# xg_vitals_found <- unique(results$ids[which(results$xg_preds.vitals == 1 & results$y )])
# lg_vitals_only <- setdiff(lg_vitals_found,c(rf_vitals_found,xg_vitals_found))
# xg_labs_only <- setdiff(xg_labs_found,c(lg_labs_found,rf_labs_found))
# 
# rf_ecg_found <- unique(results$ids[which(results$rf_preds.ecg == 1 & results$y)])
# lg_ecg_found <- unique(results$ids[which(results$lg_preds.ecg == 1 & results$y)])
# xg_ecg_found <- unique(results$ids[which(results$xg_preds.ecg == 1 & results$y )])
# lg_ecg_only <- setdiff(lg_ecg_found,c(rf_ecg_found,xg_ecg_found))
# xg_labs_only <- setdiff(xg_labs_found,c(lg_labs_found,rf_labs_found))

#write.csv(results, "super_learner_preds_results.csv", row.names=FALSE)

super_data <- data.frame(id = test_ids,y = test_y,rf_probs.labs = rf_probs.labs,rf_probs.vitals = rf_probs.vitals,rf_probs.ecg = rf_probs.ecg,lg_probs.vitals = lg_probs.vitals,lg_probs.ecg = lg_probs.ecg,xg_probs.labs = xg_probs.labs,xg_probs.vitals = xg_probs.vitals,xg_probs.ecg = xg_probs.ecg,
                         nb_probs.labs = nb.probs.labs,nb.probs.vitals = nb.probs.vitals,nb.probs.ecg = nb.probs.ecg,lg_probs.labs = lg_probs.labs,lg_probs.labs.ecg = lg_probs.labs.ecg,
                         rf_probs.labs.vitals = rf_probs.labs.vitals,rf_probs.vitals.ecg = rf_probs.vitals.ecg,rf_probs.all = rf_probs.all, rf_preds.labs.ecg = rf_preds.labs.ecg,
                         lg_probs.labs.vitals = lg_probs.labs.vitals,lg_probs.vitals.ecg = lg_probs.vitals.ecg,lg_probs.all = lg_probs.all,
                         xg_probs.labs.vitals = xg_probs.labs.vitals,xg_probs.vitals.ecg = xg_probs.vitals.ecg,xg_probs.all = xg_probs.all,
                         nb.probs.labs.ecg = nb.probs.labs.ecg,nb.probs.labs.vitals = nb.probs.labs.vitals,nb.probs.vitals.ecg= nb.probs.vitals.ecg)
super_data$y[which(super_data$y == 'TRUE')] <- 1
super_data$y[which(super_data$y == 'FALSE')] <- 0
write.csv(super_data, "super_learner_probs_results_all_combos.csv", row.names=FALSE)
library(caret)
library(MLmetrics)
super_preds = c()
test_y <- c()
super_probs_rf = c()
super_preds_rf = c()
super_probs = c()
test_ids = c()
for(i in 1:5){
  if(i == 1){
    ids = t1
    train_true <- c(t2,t3,t4,t5)
    nulls = c(nt2,nt3,nt4,nt5)
    test_null <- nt1
  }
  if(i == 2){
    ids = t2
    nulls = c(nt1,nt3,nt4,nt5)
    train_true <- c(t1,t3,t4,t5)
    test_null <- nt2
  }
  if(i == 3){
    ids = t3
    train_true <- c(t2,t1,t4,t5)
    nulls = c(nt1,nt2,nt4,nt5)
    test_null <- nt3
  }
  if(i == 4){
    ids = t4
    train_true <- c(t2,t3,t1,t5)
    nulls = c(nt1,nt3,nt2,nt5)
    test_null <- nt4
  }
  if(i == 5){
    ids = t5
    train_true <- c(t2,t3,t4,t1)
    nulls = c(nt1,nt3,nt4,nt2)
    test_null <- nt5
  }
  train <- subset(super_data,(id %in% train_true))
  train <- rbind(train,subset(super_data,id %in% null_training))
  test <- subset(super_data,id %in% ids)
  num_1 <- length(test$y[test$y == 1])
  test <- rbind(test,subset(super_data,id %in% test_null))
  #super.lg.mod <- glm(as.factor(y) ~.,data = train[,c(2,6,7,14,22)],family = 'binomial')
  super.lg.mod <- glm(as.factor(y) ~.-id,data = train,family = 'binomial')
  super_mod_rf <- glm(as.factor(y) ~.,data = train[,c(2,3:5,18)],family = 'binomial')

  num_1 <- length(test$y[test$y == 1])
  probslg <- predict(super.lg.mod,test, type = "response")
  super_probs <- c(super_probs,probslg)
  index <- order(probslg, decreasing=TRUE)[1:num_1][num_1]
  predslg <- rep(0,length(test$y))
  predslg[which(probslg >= probslg[index])] <- 1
  super_preds <- c(super_preds,predslg)

  probslg <- predict(super_mod_rf,test, type = "response")
  super_probs_rf <- c(super_probs_rf,probslg)
  index <- order(probslg, decreasing=TRUE)[1:num_1][num_1]
  predslg <- rep(0,length(test$y))
  predslg[which(probslg >= probslg[index])] <- 1
  super_preds_rf <- c(super_preds_rf,predslg)
  
  test_y <- c(test_y,test$y)
  test_ids <- c(test_ids,test$id)
  print(i)
}


F1_Score(y_pred = super_preds, y_true = test_y, positive = "1")
#0.146 with all 9, .145, with all combos it jumps up to .267 
#.12 with all but rf models

F1_Score(y_pred = super_preds_rf, y_true = test_y, positive = "1")
#.24

auc(test_y,super_probs)
#.7611 all 9, .7614, .8 with all
#.75 with all but rf
#.80 with just rf
#rf labs only had an AUC of .689

cor(super_probs,super_probs_rf)
#.4

TP = unique(test_ids[which(super_preds == 1 & test_y == 1)])
FP = unique(test_ids[which(super_preds == 1 & test_y == 0)])
TP = unique(c(TP,intersect(true_ids,FP)))
#157
FP = setdiff(FP,true_ids)
#397
TN = setdiff(null_ids,FP)
#7296
FN = setdiff(true_ids,TP)
#207

#RF Super Model
TP = unique(test_ids[which(super_preds_rf == 1 & test_y == 1)])
FP = unique(test_ids[which(super_preds_rf == 1 & test_y == 0)])
TP = unique(c(TP,intersect(true_ids,FP)))
#157
FP = setdiff(FP,true_ids)
#314
TN = setdiff(null_ids,FP)
#7379
FN = setdiff(true_ids,TP)
#207

#All Logisitic Regression
TP = unique(results$ids[which(lg_preds.all == 1 & results$y == 1)])
FP = unique(results$ids[which(lg_preds.all == 1 & results$y == 0)])
TP = unique(c(TP,intersect(true_ids,FP)))
#123
FP = setdiff(FP,true_ids)
#399
TN = setdiff(null_ids,FP)
#7294
FN = setdiff(true_ids,TP)
#241

summary(super.lg.mod)
# > summary(super.lg.mod)
# 
# Call:
#   glm(formula = as.factor(y) ~ . - ids, family = "binomial", data = train)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.2725  -0.1447  -0.1026  -0.0764   3.6801  
# 
# Coefficients:
#                   Estimate Std. Error  z value Pr(>|z|)    
#   (Intercept)     -6.97840    0.04203 -166.041   <2e-16 ***
#   rf_probs.labs    2.91210    0.07645   38.089   <2e-16 ***
#   rf_probs.vitals  1.37542    0.10786   12.752   <2e-16 ***
#   rf_probs.ecg     2.03760    0.12595   16.178   <2e-16 ***
#   lg_probs.labs    0.06868    0.09992    0.687   0.4919    
#   lg_probs.vitals  1.86611    0.12262   15.218   <2e-16 ***
#   lg_probs.ecg     0.26753    0.12599    2.124   0.0337 *  
#   xg_probs.labs   -0.03853    0.07289   -0.529   0.5970    
#   xg_probs.vitals -0.01550    0.12638   -0.123   0.9024    
#   xg_probs.ecg    -0.05040    0.11328   -0.445   0.6564    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 59202  on 522125  degrees of freedom
# Residual deviance: 52230  on 522116  degrees of freedom
# AIC: 52250
# 
# Number of Fisher Scoring iterations: 8

# Call:
#   glm(formula = as.factor(y) ~ . - ids, family = "binomial", data = train)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.7638  -0.1261  -0.0936  -0.0778   3.7820  
# 
# Coefficients:
#   Estimate Std. Error  z value Pr(>|z|)    
# (Intercept)          -6.106785   0.047506 -128.547  < 2e-16 ***
#   rf_probs.labs         3.917515   0.148426   26.394  < 2e-16 ***
#   rf_probs.vitals       0.569847   0.182785    3.118  0.00182 ** 
#   rf_probs.ecg          0.835991   0.168695    4.956 7.21e-07 ***
#   lg_probs.vitals      -1.143755   0.873011   -1.310  0.19015    
# lg_probs.ecg         -0.567905   0.259424   -2.189  0.02859 *  
#   xg_probs.labs        -1.466734   0.102757  -14.274  < 2e-16 ***
#   xg_probs.vitals       0.245580   0.170065    1.444  0.14873    
# xg_probs.ecg         -0.109465   0.151779   -0.721  0.47078    
# nb_probs.labs        -0.937281   0.054619  -17.160  < 2e-16 ***
#   nb.probs.vitals       0.004694   0.092744    0.051  0.95964    
# nb.probs.ecg          0.607209   0.098341    6.175 6.64e-10 ***
#   rf_probs.labs.vitals  3.198043   0.230388   13.881  < 2e-16 ***
#   rf_probs.vitals.ecg   1.136983   0.211417    5.378 7.54e-08 ***
#   rf_probs.all          2.679875   0.244978   10.939  < 2e-16 ***
#   rf_preds.labs.ecg     0.734633   0.055080   13.338  < 2e-16 ***
#   lg_probs.labs.vitals  1.281334   0.880455    1.455  0.14558    
# lg_probs.vitals.ecg   1.318670   0.895457    1.473  0.14085    
# lg_probs.all         -0.557055   0.892609   -0.624  0.53258    
# xg_probs.labs.vitals -1.367765   0.113520  -12.049  < 2e-16 ***
#   xg_probs.vitals.ecg   0.093517   0.160502    0.583  0.56013    
# xg_probs.all          0.030009   0.105739    0.284  0.77656    
# xg_preds.labs.ecg    -0.521619   0.077066   -6.768 1.30e-11 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 57769  on 461789  degrees of freedom
# Residual deviance: 45156  on 461767  degrees of freedom
# AIC: 45202
# 
# Number of Fisher Scoring iterations: 8

library(glmnet)
model <- glmnet(as.matrix(train)[,3:28], as.matrix(train)[,2], family = "binomial", alpha = 1, lambda = .0001)
coef(model)


super_roc = roc(test_y~super_probs)
rf_ecg_roc = roc(results$y,super_data$rf_probs.ecg)
plot(super_roc)
plot(rf_ecg_roc,add = TRUE)

super_roc <- roc_plot(super_probs,test_y)
rf_labs_roc <- roc_plot(rf_probs.labs,results$y)
rf_ecg_roc <- roc_plot(rf_probs.ecg,results$y)
plot(super_roc$FPR,super_roc$TPR,type = 'l')
lines(roc1$FPR,roc1$TPR,type = 'l',col = 'blue')
lines(rf_ecg_roc$FPR,lg_vitals_roc$TPR,type = 'l',col = 'black')

roc_plot <- function(probs,label){
  label <- label[order(-probs)]
  probs <- probs[order(-probs)]
  TPR = c(0)
  FPR = c(0)
  trues <- length(label[which(label == 1)])
  for(i in seq(1,trues*2,by = 25)){
    preds <- rep(0,length(probs))
    preds[1:i] <- 1
    TP <- length(preds[which(preds == 1 & label == 1)])
    FN <- length(preds[which(preds == 0 & label == 1)])
    FP <- length(preds[which(preds == 1 & label == 0)])
    TN <- length(preds[which(preds == 0 & label == 0)])
    TPR <- c(TPR,TP/(TP+FN))
    FPR <- c(FPR,FP/(FP+TN))
  }
  return(list("TPR" = TPR,"FPR" = FPR))
}
full_roc_plot <- function(probs,label){
  label <- label[order(-probs)]
  TPR = c(0)
  FPR = c(0)
  trues <- length(label)
  for(i in seq(1,trues,by = 1000)){
    preds <- rep(0,length(probs))
    preds[1:i] <- 1
    TP <- length(preds[which(preds == 1 & label == 1)])
    FN <- length(preds[which(preds == 0 & label == 1)])
    FP <- length(preds[which(preds == 1 & label == 0)])
    TN <- length(preds[which(preds == 0 & label == 0)])
    TPR <- c(TPR,TP/(TP+FN))
    FPR <- c(FPR,FP/(FP+TN))
  }
  return(list("TPR" = TPR,"FPR" = FPR))
}
roc1 <- roc_plot(super_probs,test_y)
auc(roc1$TPR,roc1$FPR)
#.003

#super roc area 
#.0025
auc <- function(TPR, FPR){
  # inputs already sorted, best scores first 
  dFPR <- c(diff(FPR), 0)
  dTPR <- c(diff(TPR), 0)
  sum(TPR * dFPR) + sum(dTPR * dFPR)/2
}

roc1 = roc(test_y,super_probs)
plot(roc1)
roc2 = roc(results$y,rf_probs.labs)
lines(roc2,type = 'l', col = "red")


plot(rf_labs_roc$FPR,rf_labs_roc$TPR,type = 'l', xlab="False Positive Rate", ylab="True Positive Rate")
lines(roc1$FPR,roc1$TPR,type = 'l',col = 'darkgreen')
legend("bottomright", c("Random Forest Labs", "Logistic Regression Vitals","XG Boost Vitals"), 
       col = c("black", "blue","darkgreen"),
       pch = c(19, 19,19), cex = 0.8
)
plot(roc1$FPR,roc1$TPR,type = 'l',xlab="False Positive Rate", ylab="True Positive Rate",col = "darkgreen")
lines(roc1$FPR,roc1$TPR,type = 'l',col = 'blue')

score1 <- function(preds,label){
  TP = length(preds[which(preds == 1 & label == 1)])
  FP = length(preds[which(preds == 1 & label == 0)])
  TN = length(preds[which(preds == 0 & label == 0)])
  FN = length(preds[which(preds == 1 & label == 0)])
  return(min(TP/(TP+FN),TP/(TP+FP)))
}


import.ecg$mean <- rowMeans(import.ecg[1:5])
import.labs$mean <- rowMeans(import.labs[1:5])
import.vitals$mean <- rowMeans(import.vitals[1:5])
import.ecg <- import.ecg[order(-import.ecg$mean),]

import.ecg2 <- import.labs[,c(0,6,1)]
