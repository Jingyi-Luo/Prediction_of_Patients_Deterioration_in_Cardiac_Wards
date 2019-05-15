library(RANN)
library(randomForest)
library(caret)
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

#load('C:\\Users\\x1\\Documents\\MSDS\\Capstone\\Data\\capstone_data_o2_filled.RData')
#Read in data fill in missing data
all_data <- readr::read_csv('moss_plos_one_data.csv')
all_data$O2.Flow[is.na(all_data$O2.Flow)] <- median(all_data$O2.Flow,na.rm = TRUE)
all_data$SODIUM[is.na(all_data$SODIUM)] <- median(all_data$SODIUM,na.rm = TRUE)

#find all true events
true_data <- all_data[which(all_data$y == TRUE),]

true_ids <- as.vector(unique(true_data[['id']]))
null_ids <- setdiff(as.vector(unique(all_data[['id']])),true_ids)

#true_data <- true_data[seq(1,nrow(true_data),4),]
#all_data <- all_data <- all_data[seq(1,nrow(all_data),4),]

t5 <- true_ids[296:365]#367.366
ids = t5
train <- subset(true_data,!(id %in% ids))

#add synthetic samples
train <- rbind(train,generate_syn_data(train,1))#should be about 38210

#average case has 270 observations so roughly match events to non-events by changing
#second number 

null_training <- sample(null_ids,140)#70 for no SMOTE

#add null samples to training total data set
train <- rbind(train,subset(all_data,id %in% null_training))
#remove demographic information
train <- train[,c(10,16:61)]#,63:64)]


#fit random forest model
test.rf=randomForest(as.factor(y) ~ .,data = train,ntree=200)

lg.mod <- glm(as.factor(y) ~ .,data = train,family = 'binomial')

rf_predictions <- predict(test.rf,all_data,type = 'prob')[,2]
log_reg_predictions <- predict(lg.mod,all_data,type = 'response')
all_data['rf_predictions'] = rf_predictions
all_data['log_reg_predictions'] = log_reg_predictions




#Take one observation per hour rather than 4
all_data <- all_data[seq(1,nrow(all_data),4),]
patient7019 <- all_data[which(all_data$id == 7019),]

#average observation rate
avg_risk <- length(all_data$id[which(all_data$y == TRUE)]) / length(all_data$id)
vari_names <- names(all_data)[c(16:60,63,64)]
all_data <- all_data[-which(all_data$id == 7019),]
risk2 <- rep(0,47*85)
count <- 0
#this for loop finds relative risk for all observations in patients history
#for all variabels
for(col in vari_names){
  
  nearest <- nn2(all_data[[col]],query = patient7019[[col]],k = 10000) 
  nearest <- as.data.frame(nearest$nn.idx)
  for(obs in 1:nrow(patient7019)){
    risk2[count] <- sum(all_data$y[as.integer(nearest[obs,])])/10000
    count = count + 1
    
  }
  print(col)
}
#Make x,y for event map labels
x <- -81:3
y <- vari_names
data <- expand.grid(X=x, Y=y)
data$Z <- risk2 / avg_risk #calculate relative risk

## Make actual event map
#description below about how to change colors in event map
levelplot(Z ~ X*Y, data=data  , xlab="Hour before Event" , ylab = 'Variable',
          col.regions =hsv(c(rep(.66666,6),rep(1,12)), c(rep(0,6),seq(.25,1,length.out = 12)) , 1)   , main="")


######
#c(rep(.66666,6),rep(1,12)) this portion controls color of event map, for black background need to change .6666 to black color number
#i just dont know what number that is
#c(rep(0,6),seq(.25,1,length.out = 12)) this portion controls shading of event map closer to 1 = darker, also need to make 0 in first rep()
#a darker shade like 1, but only if i can find black color number in first section
