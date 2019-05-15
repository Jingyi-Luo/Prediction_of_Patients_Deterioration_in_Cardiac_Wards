#SMOTE Implementation
library(RANN)#for knn SMOTE stuff
library(plyr)
load('C:\\Users\\x1\\Documents\\Capstone\\Data\\data_non_events_sampled.RData')

#make data frame of variables to be used in knn can't use all 48 so choose important 4
true_data_test <- data.frame(true_data$O2.Flow,true_data$edrk,true_data$WHITE.BLOOD.CELL.COUNT,true_data$hr)

#scale the data so that k-nn isn't based off nominal differnces but distance from mean instead
scaled.dat <- scale(true_data_test) 

#finds 5 nearest neighbors for all points, first is always itself
nearest <- nn2(scaled.dat,k = 6) 
nearest <- as.data.frame(nearest)

#checks to see if one of nearest neighbors is from a different patient
nearest$test <-  ((true_data[nearest$nn.idx.2,]$id != true_data[nearest$nn.idx.1,]$id)
                 |(true_data[nearest$nn.idx.3,]$id != true_data[nearest$nn.idx.1,]$id)
                 |(true_data[nearest$nn.idx.4,]$id != true_data[nearest$nn.idx.1,]$id)
                 |(true_data[nearest$nn.idx.5,]$id != true_data[nearest$nn.idx.1,]$id)
                 |(true_data[nearest$nn.idx.6,]$id != true_data[nearest$nn.idx.1,]$id)
                 )

sum(nearest$test)/length(nearest$test) #62.2% of them have neighbor from another patient

#data frame of nearest neighbor for each point in true data makes generating syntheic data easier
nearest_neighbor <- true_data[nearest$nn.idx.2,]

#synthtic data = (diff)*U(0,1) + old data where diff is difference between data and its nearest neighbor
synthetic_data <- (nearest_neighbor[,16:62]-true_data[,16:62])*runif(length(16:62),min = 0, max =1)+true_data[,16:62]
synthetic_data['y'] <- TRUE

#save synthetic data
saveRDS(synthetic_data, file = "syn_data_o2_hr_edrk_blood.rds")
synthetic_data <- readRDS(file ='C:\\Users\\x1\\Documents\\Capstone\\Data\\syn_data_o2_hr_edrk_blood.rds' )

load('C:\\Users\\x1\\Documents\\Capstone\\Data\\capstone_data_o2_filled.RData')

#see if this improved logistic regression c stat
all_data_with_syn <- rbind.fill(all_data,synthetic_data)

#Look at combined VS and Lab Results this time training on synthetic data as well as the real data
rm(all_data)

vs_lab_mod <- glm(y~WHITE.BLOOD.CELL.COUNT+BLOOD.UREA.NITROGEN+AST.GOT+PLATELET.COUNT+GLUCOSE+PCO2+POTASSIUM+SODIUM+CO2+Pulse + O2.Flow + Resp + SpO2 + SBP + Glasgow.Coma.Scale.Total,data = all_data_with_syn,family = 'binomial')

summary(vs_lab_mod)
# glm(formula = y ~ WHITE.BLOOD.CELL.COUNT + BLOOD.UREA.NITROGEN + 
#       AST.GOT + PLATELET.COUNT + GLUCOSE + PCO2 + POTASSIUM + SODIUM + 
#       CO2 + Pulse + O2.Flow + Resp + SpO2 + SBP + Glasgow.Coma.Scale.Total, 
#     family = "binomial", data = all_data_with_syn)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.5099  -0.2141  -0.1693  -0.1380   4.1774  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)               4.129e+00  2.891e-01   14.28   <2e-16 ***
#   WHITE.BLOOD.CELL.COUNT    5.366e-02  1.176e-03   45.63   <2e-16 ***
#   BLOOD.UREA.NITROGEN       9.912e-03  2.094e-04   47.33   <2e-16 ***
#   AST.GOT                   4.991e-03  1.304e-04   38.27   <2e-16 ***
#   PLATELET.COUNT           -2.072e-03  5.417e-05  -38.24   <2e-16 ***
#   GLUCOSE                   2.331e-03  8.288e-05   28.12   <2e-16 ***
#   PCO2                      4.133e-02  1.592e-03   25.96   <2e-16 ***
#   POTASSIUM                 1.296e-01  1.028e-02   12.61   <2e-16 ***
#   SODIUM                   -1.242e-02  1.177e-03  -10.55   <2e-16 ***
#   CO2                      -3.406e-02  1.210e-03  -28.15   <2e-16 ***
#   Pulse                     2.249e-02  2.773e-04   81.10   <2e-16 ***
#   O2.Flow                   2.937e-01  3.628e-03   80.95   <2e-16 ***
#   Resp                      7.114e-02  1.400e-03   50.80   <2e-16 ***
#   SpO2                     -7.456e-02  1.814e-03  -41.11   <2e-16 ***
#   SBP                      -2.851e-03  2.198e-04  -12.97   <2e-16 ***
#   Glasgow.Coma.Scale.Total -3.226e-01  6.999e-03  -46.09   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 462154  on 2241838  degrees of freedom
# Residual deviance: 425874  on 2241823  degrees of freedom
# AIC: 425906
# 
# Number of Fisher Scoring iterations: 7

#try and estimate the c statistic by picking one event one non event 
#predict which one is event 

#my laptop can't hold too many large data sets so clear out extra copy
rm(all_data_with_syn)

load('C:\\Users\\x1\\Documents\\Capstone\\Data\\capstone_data_o2_filled.RData')
count = 0 
for(i in 1:10000){
  test <- TRUE
  while(test == TRUE){
    null_event <- all_data[sample(nrow(all_data),1), ]
    test <- null_event$y
  }
  test_event <- true_data[sample(nrow(true_data),1),]
  p0 <- predict(vs_lab_mod,null_event,type= "response")
  p1 <- predict(vs_lab_mod,test_event,type= "response")
  if(p1 > p0){
    count = count + 1
  }
}
count / 10000
#.7178 is a little higher than the previous .716 might just be chance though


