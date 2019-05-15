#Capstone Data Initial Analysis
library(dplyr)#to read in large file
library(RANN)#for knn SMOTE stuff
setwd('C:\\Users\\x1\\Documents\\Capstone\\Data')

all_data <- readr::read_csv('moss_plos_one_data.csv')
#Fill in only missing values with median
all_data$O2.Flow[is.na(all_data$O2.Flow)] <- median(all_data$O2.Flow,na.rm = TRUE)
all_data$SODIUM[is.na(all_data$SODIUM)] <- median(all_data$SODIUM,na.rm = TRUE)
#Saved R data for easier loading later
load('C:\\Users\\x1\\Documents\\Capstone\\Data\\capstone_data_o2_filled.RData')

length(unique(all_data$id))
#8105 different patients matches paper
sum(all_data$y)/length(all_data$y)
#1.08% of observations are True

hr_mod <- glm(y~hr,data = all_data,family = 'binomial')
summary(hr_mod)
#Coefficients:
#             Estimate  Std. Error z value Pr(>|z|)    
#(Intercept) -6.8605299  0.0327133 -209.72   <2e-16 ***
#hr           0.0279127  0.0003654   76.39   <2e-16 ***
#Null deviance: 263926  on 2217957  degrees of freedom
#Residual deviance: 258597  on 2217956  degrees of freedom
#AIC: 258601

#try and estimate the c statistic by picking one event one non event 
#predict which one is event 
true_data <-  subset(all_data,y == TRUE)
write.csv(true_data,'events.csv')

count = 0 
for(i in 1:10000){
test <- TRUE
while(test == TRUE){
null_event <- all_data[sample(nrow(all_data),1), ]
test <- null_event$y
}
test_event <- true_data[sample(nrow(true_data),1),]
p0 <- predict(hr_mod,null_event,type= "response")
p1 <- predict(hr_mod,test_event,type= "response")
if(p1 > p0){
  count = count + 1
}
}
count / 10000
#.634 pretty sure this is c stat seems about right given results from the paper

#Next do same as above but try and include all VS variables


VS_mod <- glm(y~Pulse + O2.Flow + Resp + SpO2 + SBP + Glasgow.Coma.Scale.Total,data = all_data,family = 'binomial')
summary(VS_mod)
#Coefficients:
#  Estimate Std. Error z value Pr(>|z|)    
#(Intercept)               4.4657442  0.2991536  14.928  < 2e-16 ***
#  Pulse                     0.0225496  0.0003847  58.620  < 2e-16 ***
#  O2.Flow                   0.3092853  0.0048790  63.392  < 2e-16 ***
#  Resp                      0.0777427  0.0019049  40.811  < 2e-16 ***
#  SpO2                     -0.0750289  0.0025621 -29.284  < 2e-16 ***
#  SBP                      -0.0019812  0.0002998  -6.608 3.89e-11 ***
#  Glasgow.Coma.Scale.Total -0.3805437  0.0092962 -40.935  < 2e-16 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
#(Dispersion parameter for binomial family taken to be 1)
#Null deviance: 255719  on 2179116  degrees of freedom
#Residual deviance: 242925  on 2179110  degrees of freedom
#AIC: 242939ance: 258597  on 2217956  degrees of freedom
#AIC: 258601

#try and estimate the c statistic by picking one event one non event 
#predict which one is event 
count = 0 
for(i in 1:10000){
  test <- TRUE
  while(test == TRUE){
    null_event <- all_data[sample(nrow(all_data),1), ]
    test <- null_event$y
  }
  test_event <- true_data[sample(nrow(true_data),1),]
  p0 <- predict(VS_mod,null_event,type= "response")
  p1 <- predict(VS_mod,test_event,type= "response")
  if(p1 > p0){
    count = count + 1
  }
}
count / 10000
#.6854 again think its roughly c stat is very close to the .687 in paper

#Next do same as above but try and include all lab variables
all_data$SODIUM[is.na(all_data$SODIUM)] <- median(all_data$SODIUM,na.rm = TRUE)

lab_mod <- glm(y~WHITE.BLOOD.CELL.COUNT+BLOOD.UREA.NITROGEN+AST.GOT+PLATELET.COUNT+GLUCOSE+PCO2+POTASSIUM+SODIUM+CO2,data = all_data,family = 'binomial')
summary(lab_mod)
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)            -5.692e+00  2.643e-01  -21.54  < 2e-16 ***
#   WHITE.BLOOD.CELL.COUNT  7.969e-02  1.626e-03   49.00  < 2e-16 ***
#   BLOOD.UREA.NITROGEN     1.101e-02  2.931e-04   37.57  < 2e-16 ***
#   AST.GOT                 5.518e-03  1.775e-04   31.09  < 2e-16 ***
#   PLATELET.COUNT         -2.054e-03  7.585e-05  -27.08  < 2e-16 ***
#   GLUCOSE                 2.554e-03  1.149e-04   22.22  < 2e-16 ***
#   PCO2                    4.528e-02  2.969e-03   15.25  < 2e-16 ***
#   POTASSIUM               7.518e-02  1.468e-02    5.12 3.06e-07 ***
#   SODIUM                 -1.014e-02  1.634e-03   -6.21 5.29e-10 ***
#   CO2                    -2.750e-02  1.707e-03  -16.11  < 2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 259568  on 2210284  degrees of freedom
# Residual deviance: 252915  on 2210275  degrees of freedom
# (7673 observations deleted due to missingness)
# AIC: 252935
# 
# Number of Fisher Scoring iterations: 7

#try and estimate the c statistic by picking one event one non event 
#predict which one is event 
count = 0 
for(i in 1:10000){
  test <- TRUE
  while(test == TRUE){
    null_event <- all_data[sample(nrow(all_data),1), ]
    test <- null_event$y
  }
  test_event <- true_data[sample(nrow(true_data),1),]
  p0 <- predict(lab_mod,null_event,type= "response")
  p1 <- predict(lab_mod,test_event,type= "response")
  if(p1 > p0){
    count = count + 1
  }
}
count / 10000
#.622 again think its roughly c stat is very close to the .629 in paper


#Look at combined VS and Lab Results
vs_lab_mod <- glm(y~WHITE.BLOOD.CELL.COUNT+BLOOD.UREA.NITROGEN+AST.GOT+PLATELET.COUNT+GLUCOSE+PCO2+POTASSIUM+SODIUM+CO2+Pulse + O2.Flow + Resp + SpO2 + SBP + Glasgow.Coma.Scale.Total,all_data = data,family = 'binomial')
summary(vs_lab_mod)
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)               3.810e+00  4.053e-01   9.401   <2e-16 ***
#   WHITE.BLOOD.CELL.COUNT    5.449e-02  1.644e-03  33.140   <2e-16 ***
#   BLOOD.UREA.NITROGEN       9.816e-03  2.910e-04  33.732   <2e-16 ***
#   AST.GOT                   4.851e-03  1.817e-04  26.688   <2e-16 ***
#   PLATELET.COUNT           -2.174e-03  7.666e-05 -28.353   <2e-16 ***
#   GLUCOSE                   2.406e-03  1.161e-04  20.733   <2e-16 ***
#   PCO2                      3.997e-02  2.219e-03  18.016   <2e-16 ***
#   POTASSIUM                 1.317e-01  1.446e-02   9.109   <2e-16 ***
#   SODIUM                   -1.372e-02  1.656e-03  -8.286   <2e-16 ***
#   CO2                      -3.371e-02  1.697e-03 -19.865   <2e-16 ***
#   Pulse                     2.246e-02  3.858e-04  58.214   <2e-16 ***
#   O2.Flow                   2.930e-01  4.990e-03  58.711   <2e-16 ***
#   Resp                      7.256e-02  1.923e-03  37.744   <2e-16 ***
#   SpO2                     -7.548e-02  2.543e-03 -29.681   <2e-16 ***
#   SBP                      -2.850e-03  3.100e-04  -9.194   <2e-16 ***
#   Glasgow.Coma.Scale.Total -3.282e-01  9.563e-03 -34.322   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 263926  on 2217957  degrees of freedom
# Residual deviance: 245320  on 2217942  degrees of freedom
# AIC: 245352
# 
# Number of Fisher Scoring iterations: 8

#try and estimate the c statistic by picking one event one non event 
#predict which one is event 
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
#.716 again think its roughly c stat is very close to the .714 in paper only one thats been higher


length(unique(true_data$id))
#only 367 paper says it should be 586
#only 367 were monitored 24 hours prior so this number is correct

#sort so only 1 observation per idea to check demographics data
demo_true_data <- true_data %>% group_by(id) %>% filter(row_number(id) == 1)
median(demo_true_data$age)
#66.7 matches the 67 on paper
median(demo_true_data$LOS)
#11 matches paper
length(subset(demo_true_data,race == 'wht')$race)/length(demo_true_data$id)
#81.2% slightly lower than paper, btut firs since theirs included patients that had unmonitored events

#SMOTE Stuff
true_data_test <- data.frame(true_data$O2.Flow,true_data$Resp,true_data$BLOOD.UREA.NITROGEN,true_data$hr)

scaled.dat <- scale(true_data_test)
nearest <- nn2(scaled.dat,k = 5)
nearest[1]
nearest <- as.data.frame(nearest)
nearest$test <- ((abs(nearest$nn.idx.2-nearest$nn.idx.1)>10) | (abs(nearest$nn.idx.3-nearest$nn.idx.1)>10)| (abs(nearest$nn.idx.4-nearest$nn.idx.1)>10))
sum(nearest$test)/length(nearest$test)#most nearest neighbors are from patients with other ids

synthetic_data <- all_data[0,]
for(obs in 1:5){
  nearest_index <- nearest[1]$nn.idx[obs,2]
  new_event <- (true_data[obs,16:62] - true_data[nearest_index,16:62])*runif(length(16:62), min=0, max=1) + true_data[obs,16:62]
  new_event$y <- TRUE
  synthetic_data <- rbind(synthetic_data, new_event)
  print(obs)
}
