rm(list = ls())    # reset / remove all objects

#Install / load pertinent libraries
library(data.table)
library(glmnet)
library(randomForest)
library(ISLR)
library(MASS)
library(ggplot2)
library(class)
library(dplyr)
library(tidyverse)
library(coefplot)
library(wesanderson)
library(base)

##############################################################
### INTRODUCTION: IMPORTS, CLEANING, AND INITIAL WRANGLING ###
##############################################################

#Bring in sampled COVID-19 dataset
covid_ <- read_csv("Downloads/covid_.csv")
covid2 <- data.table(covid_)

#Regulate certain regressors by TotalPop (total population)
covid2$VotingAgeCitizenPct <- covid2$VotingAgeCitizen / covid2$TotalPop
covid2$WomenPct <- covid2$Women / covid2$TotalPop
covid2$VotingPct20 <- covid2$total_votes20 / covid2$TotalPop

##Formulate new variable voter turnout (calculated field)
covid2$VoterTurnoutPct20 <- covid2$total_votes20 / covid2$VotingAgeCitizen

##Removing impertinent variables
covid2$X1 <- NULL
covid2$deaths <- NULL
covid2$county <- NULL
covid2$Employed <- NULL
covid2$Men <- NULL
covid2$Women <- NULL
covid2$TotalPop <- NULL
covid2$VotingAgeCitizen <- NULL
covid2$votes16_Donald_Trump <- NULL
covid2$votes16_Hillary_Clinton <- NULL
covid2$votes20_Donald_Trump <- NULL
covid2$votes20_Joe_Biden <- NULL
covid2$total_votes16 <- NULL
covid2$total_votes20 <- NULL

##Log our depdendent variable (cases by county)
y = log(covid2$cases + 1) #shift to keep 0s (otherwise we lose the data as NA values)

##Additional data regarding our y
mean(y)
median(y)
sd(y)

##Here, we see that while cases has a large right tail, 
##its log roughly follows a normal distribution
hist(covid2$cases)
hist(y)

##Dimensions of the data
covid2$cases <- NULL
n = dim(covid2)[1] #n = 1,468
p = dim(covid2)[2] #p = 40

##Convert regressor to X matrix for analysis
covid2_regressors  = covid2
covid_mat = data.matrix(covid2_regressors)

##Additional data regarding our regressor matrix
apply(covid_mat, 2, 'mean') #mean of each column
apply(covid_mat, 2, 'sd') #sd of each column

##Additional Insight: Standardization (by Gaussian scaling)

###mu = as.vector(apply(covid_mat, 2, 'mean'))
###sd = as.vector(apply(covid_mat, 2, 'sd'))
###covid_mat_std   =   covid_mat
###for (i in c(1:n)){
###  covid_mat_std[i,]  =   (covid_mat_std[i,] - mu)/sd
###}
###apply(covid_mat_std, 2, 'mean') #should mirror a zero vector
###apply(covid_mat_std, 2, 'sd') #should mirrora vector of 1s

##############################################################
################# NUMBER 3: 100 SAMPLES ######################
##############################################################

##Declare / initialize values and vectors

###Setting seed, repetition count, and 80-20 split
set.seed(1)
n.train = floor(0.8 * n)
n.test = n - n.train
run_amt = 100

###lasso R-Squared vectors (initialize as a zero vector)
Rsq.train.lasso = rep(0,run_amt)
Rsq.test.lasso = rep(0,run_amt) 

###elastic net R-Squared vectors (initialize as a zero vector)
Rsq.train.en = rep(0,run_amt)
Rsq.test.en = rep(0,run_amt) 

###ridge regression R-Squared vectors (initialize as a zero vector)
Rsq.train.ridge = rep(0,run_amt)
Rsq.test.ridge = rep(0,run_amt)

###randomForest R-Squared vectors (initialize as a zero vector)
Rsq.train.rf = rep(0,run_amt)
Rsq.test.rf = rep(0,run_amt)


##Run the 100 times
for (i in c(1:run_amt)) {
 
  ##Randomly plit the data into train and test
  shuffled_indexes = sample(n)
  train = shuffled_indexes[1:n.train]
  test = shuffled_indexes[(1+n.train):n]

  X.train = covid_mat[train,]
  y.train = y[train]
  X.test = covid_mat[test,]
  y.test = y[test]
  
  ##Lasso Method
  lasso.cv = cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  lasso.fit = glmnet(X.train, y.train, alpha = 1, lambda = lasso.cv$lambda.min)
  y.train.hat = predict(lasso.fit, newx = X.train, type = "response") 
  y.test.hat = predict(lasso.fit, newx = X.test, type = "response") 
  Rsq.train.lasso[i] = 1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.lasso[i] = 1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  ##Elastic-net Method 
  a = 0.5 #use alpha = 0.5 for elastic-net
  en.cv = cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  en.fit = glmnet(X.train, y.train, alpha = a, lambda = en.cv$lambda.min)
  y.train.hat =predict(en.fit, newx = X.train, type = "response") 
  y.test.hat= predict(en.fit, newx = X.test, type = "response") 
  Rsq.train.en[i]= 1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.en[i]=1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  ##Ridge Regression Method  
  ridge.cv=cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  ridge.fit=glmnet(X.train, y.train, alpha = 0, lambda = ridge.cv$lambda.min)
  y.train.hat=predict(ridge.fit, newx = X.train, type = "response") 
  y.test.hat=predict(ridge.fit, newx = X.test, type = "response") 
  Rsq.train.ridge[i]=1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.ridge[i]=1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  ##Random Forest Method 
  rf.fit=randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.train.hat=predict(rf.fit, X.train)
  y.test.hat= predict(rf.fit, X.test)
  Rsq.train.rf[i]= 1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.rf[i]= 1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  ##Log (for tracking process and for error handling the script)
  cat(sprintf("Fit count:%3.f| with train @ 0.8n: 
              Train-- Lasso=%.3f, En=%.3f, Ridge=%.3f, rF=%.3f
              Test--  Lasso=%.3f, En=%.3f, Ridge=%.3f, rF=%.3f\n",i, 
              Rsq.train.lasso[i], Rsq.train.en[i], Rsq.train.ridge[i], Rsq.train.rf[i],
              Rsq.test.lasso[i], Rsq.test.en[i], Rsq.test.ridge[i], Rsq.test.rf[i]))
}


##############################################################
############## NUMBER 4B: R-SQUARED BOXPLOTS #################
##############################################################

##Put R-squared values in a long format data.table
dt_Rsq = data.frame(c(rep("Train", 4*run_amt), rep("Test", 4*run_amt)), 
                    c(rep("Lasso",run_amt),rep("Elastic-net",run_amt), 
                      rep("Ridge",run_amt),rep("Random Forest",run_amt), 
                      rep("Lasso",run_amt),rep("Elastic-net",run_amt), 
                      rep("Ridge",run_amt),rep("Random Forest",run_amt)), 
                    c(Rsq.train.lasso, Rsq.train.en, Rsq.train.ridge, Rsq.train.rf, 
                      Rsq.test.lasso, Rsq.test.en, Rsq.test.ridge, Rsq.test.rf))

colnames(dt_Rsq) =  c("Type", "Method", "R2")
dt_Rsq

##Change factor level order (for unity purposes)
dt_Rsq$Method = factor(dt_Rsq$Method, levels=c("Lasso", "Elastic-net", "Ridge", "Random Forest"))
dt_Rsq$Type = factor(dt_Rsq$Type, levels=c("Train", "Test"))

##Plot boxplot using ggplot
ggplot(dt_Rsq,aes(x = Method, y = R2, fill = Method)) + geom_boxplot() + facet_wrap(~Type, ncol = 2) +
  theme(axis.text.x = element_text(hjust = 1,vjust = 0.5, size = 30/.pt, angle = 90), plot.title=element_text(hjust = 0.5)) + 
  ggtitle("Boxplots: R-Squared of Train and Test")


##############################################################
########### NUMBER 4C: CV CURVES FOR ONE SAMPLE ##############
##############################################################

##Note: We are using the 100th of the 100 since the fields are currently set to such

##Fields of 100th sample established in the prior for loop:
X.train
y.train
X.test
y.test  


#######################
######## Lasso ######## 

##Start time
ptm=proc.time()

##Cross-validate
lasso.cv = cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)

##Stop time
ptm = proc.time() - ptm
time_lasso  =ptm["elapsed"]
cat(sprintf("Run Time for Lasso: %0.3f(sec):",time_lasso))

##Plot CV Curve
plot(lasso.cv)+title("10-fold CV curve for Lasso", line = 2.5)

#######################
##### Elastic-net ##### 

a = 0.5 #use alpha = 0.5 for elastic-net

##Start time
ptm = proc.time()

##Cross-validate
en.cv =cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)

##Stop time
ptm  = proc.time() - ptm
time_en=   ptm["elapsed"]
cat(sprintf("Run Time for Elastic-net: %0.3f(sec):",time_en))

##Plot CV Curve
plot(en.cv)+title("10-fold CV curve for Elastic-net", line = 2.5)

#######################
######## Ridge ######## 

##Start time
ptm=proc.time()

##Cross-validate
ridge.cv=cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)

##Stop time
ptm=proc.time() - ptm
time_ridge= ptm["elapsed"]
cat(sprintf("Run Time for ridge: %0.3f(sec):",time_ridge))

##Plot CV Curve
plot(ridge.cv)+title("10-fold CV curve for Ridge", line = 2.5)

##############################################################
############## NUMBER 4D: RESIDUAL BOXPLOTS ##################
##############################################################

#######################
######## Lasso ######## 

##Fit method
lasso.fit = glmnet(X.train, y.train, alpha = 1, lambda = lasso.cv$lambda.min)

##Calculate y-hats
y.train.hat.lasso = predict(lasso.fit, newx = X.train, type = "response") 
y.test.hat.lasso = predict(lasso.fit, newx = X.test, type = "response") 

##Calculate residuals
Rsq.train_lasso = 1 - mean((y.train - y.train.hat.lasso)^2)/mean((y - mean(y))^2)  
Rsq.test_lasso= 1 - mean((y.test - y.test.hat.lasso)^2)/mean((y - mean(y))^2)

##Wrangling residuals
y.train.hat.lasso = as.vector(y.train.hat.lasso)
y.test.hat.lasso = as.vector(y.test.hat.lasso)

residual.lasso=data.table(c(rep("Train", n.train),rep("Test", n.test)),   
                          c(1:n), c(y.train.hat.lasso-y.train, y.test.hat.lasso-y.test))
colnames(residual.lasso) = c("Type", "Data_Row", "Residual")


#######################
##### Elastic-net ##### 

##Fit method
en.fit= glmnet(X.train, y.train, alpha = a, lambda = en.cv$lambda.min)

##Calculate y-hats
y.train.hat.en=predict(en.fit, newx = X.train, type = "response") 
y.test.hat.en=predict(en.fit, newx = X.test, type = "response") 

##Calculate residuals
Rsq.train_en=1-mean((y.train - y.train.hat.en)^2)/mean((y - mean(y))^2)  
Rsq.test_en=1-mean((y.test - y.test.hat.en)^2)/mean((y - mean(y))^2)

##Wrangling residuals
y.train.hat.en =as.vector(y.train.hat.en)
y.test.hat.en =as.vector(y.test.hat.en)

residual.en = data.table(c(rep("Train", n.train),rep("Test", n.test)),   
                         c(1:n), c(y.train.hat.en-y.train, y.test.hat.en-y.test))
colnames(residual.en) =  c("Type", "Data_Row", "Residual")


#######################
######## Ridge ######## 

##Fit method
ridge.fit=glmnet(X.train, y.train, alpha = 0, lambda = ridge.cv$lambda.min)

##Calculate y-hats
y.train.hat.ridge = predict(ridge.fit, newx = X.train, type = "response") 
y.test.hat.ridge = predict(ridge.fit, newx = X.test, type = "response")

##Calculate residuals
Rsq.train_ridge=1 - mean((y.train - y.train.hat.ridge)^2)/mean((y - mean(y))^2)  
Rsq.test_ridge=1 - mean((y.test - y.test.hat.ridge)^2)/mean((y - mean(y))^2)

##Wrangling residuals
y.train.hat.ridge =as.vector(y.train.hat.ridge)
y.test.hat.ridge =as.vector(y.test.hat.ridge)

residual.ridge = data.table(c(rep("Train", n.train),rep("Test", n.test)),  
                            c(1:n), c(y.train.hat.ridge-y.train, y.test.hat.ridge-y.test))
colnames(residual.ridge) =  c("Type", "Data_Row", "Residual")


#######################
#### Random Forest ####

##Fit method
rf.fit=randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)

##Calculate y-hats
y.train.hat.rf=predict(rf.fit, X.train)
y.test.hat.rf= predict(rf.fit, X.test)

##Calculate residuals
Rsq.train_rf= 1 - mean((y.train - y.train.hat.rf)^2)/mean((y - mean(y))^2)  
Rsq.test_rf= 1 - mean((y.test - y.test.hat.rf)^2)/mean((y - mean(y))^2)

##Wrangling residuals
y.train.hat.rf =as.vector(y.train.hat.rf)
y.test.hat.rf =as.vector(y.test.hat.rf)

residual.rf = data.table(c(rep("Train", n.train),rep("Test", n.test)),  
                         c(1:n), c(y.train.hat.rf-y.train, y.test.hat.rf-y.test))
colnames(residual.rf) =  c("Type", "Data_Row", "Residual")

##Consolidate all residuals in a long format data.table
residual.dt = data.frame(c(rep("Lasso",n), rep("Elastic-net",n), rep("Ridge",n), rep("Random Forest",n)),
                         rbind(residual.lasso, residual.en, residual.ridge, residual.rf))

colnames(residual.dt) = c("Method", "Type", "Data_Row", "Residual")


##Change factor level order (for unity purposes)
residual.dt$Method = factor(residual.dt$Method, levels = c("Lasso", "Elastic-net", "Ridge", "Random Forest"))
residual.dt$Type = factor(residual.dt$Type, levels = c("Train", "Test"))


##Plot boxplot using ggplot
ggplot(residual.dt, aes(x = Method, y = Residual, fill = Method)) + geom_boxplot() + facet_wrap(~Type, ncol = 2) + 
  theme(axis.text.x = element_text(angle = 90,hjust = 1,vjust = 0.5, size = 30/.pt)) +
  ggtitle("Train and Test Residuals") + theme(plot.title = element_text(hjust = .5,size = 40/.pt))


## Note: For number 5, we note the bullets as letter  ##
##### For example, the first bullet will be notated ####
##################### as Number 5A #####################


##############################################################
################# NUMBER 5A: FIT ALL DATA ####################
##############################################################

## Note: We will also track time for the second bullet

#######################
######## Lasso ######## 

##Start Time
ptm=proc.time()

##Cross-validate / fit method
lasso.cv = cv.glmnet(covid_mat, y, alpha = 1, nfolds = 10)
lasso.fit = glmnet(covid_mat, y, alpha = 1, lambda = lasso.cv$lambda.min)

##Stop Time
ptm = proc.time() - ptm
time_lasso = ptm["elapsed"]
cat(sprintf("Run Time for Lasso: %0.3f(sec):",time_lasso))

##Additional Insight: Calculating R-squared
y.hat.lasso = predict(lasso.fit, newx = covid_mat, type = "response") 
Rsq_lasso = 1 - mean((y - y.hat.lasso)^2) / mean((y - mean(y))^2)

#######################
##### Elastic-net ##### 

a = 0.5 #use alpha = 0.5 for elastic-net

##Start Time
ptm = proc.time()

##Cross-validate / fit method
en.cv = cv.glmnet(covid_mat, y, alpha = a, nfolds = 10)
en.fit = glmnet(covid_mat, y, alpha = a, lambda = en.cv$lambda.min)

##Stop Time
ptm= proc.time() - ptm
time_en = ptm["elapsed"]
cat(sprintf("Run Time for elastic-net: %0.3f(sec):",time_en))

##Additional Insight: Calculating R-squared
y.hat.en = predict(en.fit, newx = covid_mat, type = "response") 
Rsq_en = 1 - mean((y - y.hat.en)^2) / mean((y - mean(y))^2)  


#######################
######## Ridge ######## 

##Start Time
ptm=proc.time()

##Cross-validate / fit method
ridge.cv = cv.glmnet(covid_mat, y, alpha = 0, nfolds = 10)
ridge.fit = glmnet(covid_mat, y, alpha = 0, lambda = ridge.cv$lambda.min)

##Stop Time
ptm=proc.time() - ptm
time_ridge= ptm["elapsed"]
cat(sprintf("Run Time for ridge: %0.3f(sec):",time_ridge))

##Additional Insight: Calculating R-squared
y.hat.ridge = predict(ridge.fit, newx = covid_mat, type = "response") 
Rsq_ridge = 1 - mean((y - y.hat.ridge)^2) / mean((y - mean(y))^2)  


#######################
#### Random Forest ####

##Start Time
ptm=proc.time()

##Cross-validate / fit method
rf.fit=randomForest(covid_mat, y, mtry = sqrt(p), importance = TRUE)

##Stop Time
ptm = proc.time() - ptm
time_rf = ptm["elapsed"]
cat(sprintf("Run Time for Random Forest: %0.3f(sec):",time_rf))

##Additional Insight: Calculating R-squared
y.hat.rf=predict(rf.fit, covid_mat)
Rsq_rf= 1-mean((y - y.hat.rf)^2)/mean((y - mean(y))^2)  


##############################################################
############## NUMBER 5B: PERFORMANCE VS TIME ################
##############################################################


## Note: We are using the Test R-squared vector from the for loop of 3
Rsq.test.lasso
Rsq.test.en
Rsq.test.ridge
Rsq.test.rf

##Constructing 90% Confidence Intervals for R-squared
###for the 100 repetitions
lasso.Rsq.ci = t.test(Rsq.test.lasso, conf.level = 0.9)
en.Rsq.ci = t.test(Rsq.test.en, conf.level = 0.9)
ridge.Rsq.ci = t.test(Rsq.test.ridge, conf.level = 0.9)
rf.Rsq.ci = t.test(Rsq.test.rf, conf.level = 0.9)
lasso.Rsq.ci$conf.int[1:2]
en.Rsq.ci$conf.int[1:2]
ridge.Rsq.ci$conf.int[1:2]
rf.Rsq.ci$conf.int[1:2]

## Note: We are using the time values from the 5A run
time_lasso
time_en
time_ridge
time_rf

#Consolidate time and performance data in long format data.table
time_perf.dt = data.table(rep(c("Lasso", "Elastic-net", "Ridge", "Random Forest"),2) ,
                          rep(c(time_lasso, time_en, time_ridge, time_rf),2),
                          rep(c("Lower Bound", "Upper Bound"),4),
                          c(lasso.Rsq.ci$conf.int[1], en.Rsq.ci$conf.int[1], ridge.Rsq.ci$conf.int[1], rf.Rsq.ci$conf.int[1],
                            lasso.Rsq.ci$conf.int[2], en.Rsq.ci$conf.int[2], ridge.Rsq.ci$conf.int[2], rf.Rsq.ci$conf.int[2]))

colnames(time_perf.dt) = c("Method", "Time", "Bound Type", "R2")

##Plot Time against 90% Confidence Intervals for R-squared
ggplot(time_perf.dt, aes(x = Time, y = R2, color = Method)) + geom_point() + ylab("R-squared") + xlab("Time (in seconds)") + 
  ggtitle("Performance vs. Time Complexity") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))


##############################################################
############### NUMBER 5C: VARIABLE ANALYSIS #################
##############################################################

##Estimated Coefficients for Lasso, Elastic-net, and Ridge
lasso_beta = data.table(as.character(c(1:p)), as.vector(lasso.fit$beta))
en_beta = data.table(as.character(c(1:p)), as.vector(en.fit$beta))
ridge_beta = data.table(as.character(c(1:p)), as.vector(ridge.fit$beta))

##Variable Importance for Random Forest
rf_importance = data.table(as.character(c(1:p)), as.vector(rf.fit$importance[,1]))

##Rename columns for uniformity 
colnames(lasso_beta) = c("param", "value")
colnames(en_beta) = c("param", "value")
colnames(ridge_beta) = c("param", "value")
colnames(rf_importance) = c("param", "value")

##Additional Insight: Individual Method Bar Plots
###rf = ggplot(rf_importance, aes(x = param , y = value)) + geom_col() + theme(axis.text.x = element_text(angle=90,hjust = 1,vjust = 0.5))
###las = ggplot(lasso_beta, aes(x = param , y=value)) + geom_col() + theme(axis.text.x = element_text(angle=90,hjust = 1,vjust = 0.5))
###en = ggplot(en_beta, aes(x = param , y=value)) + geom_col() + theme(axis.text.x = element_text(angle = 90,hjust = 1,vjust = 0.5))
###ri = ggplot(ridge_beta, aes(x = param , y=value)) + geom_col() + theme(axis.text.x = element_text(angle=90,hjust = 1,vjust = 0.5))

##Ordering all methods by Elastic-net's estimated coefficients
en_beta$param = factor(en_beta$param, levels = en_beta$param[order(en_beta$value, decreasing= TRUE)])
rf_importance$param = factor(rf_importance$param, levels = en_beta$param[order(en_beta$value, decreasing= TRUE)])
lasso_beta$param = factor(lasso_beta$param, levels = en_beta$param[order(en_beta$value, decreasing= TRUE)])
ridge_beta$param = factor(ridge_beta$param, levels = en_beta$param[order(en_beta$value, decreasing= TRUE)])


##Consolidate all residuals in a long format data.table
rf_importance$Method = "Random Forest" 
lasso_beta$Method = "Lasso" 
en_beta$Method = "Elastic-net" 
ridge_beta$Method = "Ridge" 

param_dt <- data.table(rbind(en_beta, lasso_beta, ridge_beta, rf_importance))

colnames(param_dt) = c("Parameter" ,"Value", "Method")
param_dt

##Change factor level order (for unity purposes)
param_dt$Method = factor(param_dt$Method, levels = c("Elastic-net","Lasso", "Ridge", "Random Forest"))

##Plot barchat using ggplot (a geom_col call)
### Note: we are using "wes_palette" for a more defined color palette
### which optically distinguishes Random Forest from the other methods
ggplot(param_dt,aes(x = Parameter, y = Value, fill = Method)) + geom_col() +
  facet_wrap(~Method, nrow = 4, scales= "free_y")+
  scale_fill_manual(values=wes_palette(n=4, name="Rushmore")) + theme(axis.text.x = element_text(angle = 90,hjust = 1,vjust = 0.5, size = 30/.pt)) +
  ggtitle("Parameter Importance By Method") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

##Additional Insight: check for identifying parameters from the above barplot
###param_number = 13
###colnames(covid2_regressors)[param_number]





