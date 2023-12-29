#
# Course 8 - Project 1
#

###
### They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: 
### http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 
###

#C:\Users\RussS\Documents\Coursera\Data Science Specialization Certificate\Course 8 - ML\Project1\Data

### What you should submit
### The goal of your project is to predict the manner in which they did the exercise. 
### This is the "classe" variable in the training set. You may use any of the other variables to predict with. 
### You should create a report describing how you built your model, how you used cross validation, 
### what you think the expected out of sample error is, and why you made the choices you did. 
### You will also use your prediction model to predict 20 different test cases. 
library(dplyr)
library(caret)
###library(rpart)

###
### Load in test and training data.
###
training_df = read.csv("./Data/pml-training.csv")
testing_df = read.csv("./Data/pml-testing.csv")

### Look at data
str(training_df)
names(training_df)

### Look at counts for each classe (target variable) to see if there is any imbalanced (underrepresented) outcome values.
g <- group_by(training_df, training_df$classe)
summarise(g,count = n())


### Look at data - but it's has too many columns to view this way:
#View(training_df)
#View(testing_df)

# Remove all rows from training dataset where new_window == yes as they are not in the testing data
training_df_1 <- filter(training_df, new_window=='no') 

# Look at columns that should not be features based on the problem we are trying to solve
#View(training_df_1)

#install.packages("tidyverse")
library(tidyverse)

# get rid of all columns that are empty
training_df_2 <- discard(training_df_1,~all(is.na(.) | . == ""))
#View(training_df_2)

# remove all columns that will are not features or the outcome variable
# X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, new_window, num_window
colsToRemove <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "new_window", "num_window", "cvtd_timestamp")
training_df_final <- select(training_df_2,-c(colsToRemove))
#View(training_df_final)

# set seed to ensure that the results can be reproduced by others
set.seed(21)

### Split the training data set into a training and validation set for out of sample error testing.
inTrain <- createDataPartition(y=training_df_final$classe, p=0.7, list=FALSE)
training <- training_df_final[inTrain,]
validation <- training_df_final[-inTrain,]

# clean up test data as well
testing_df_2 <- discard(testing_df,~all(is.na(.) | . == ""))
#View(testing_df_2)

testing_df_final <- select(testing_df_2,-c(colsToRemove))
#View(testing_df_final)

# Cross validation
# apply k-fold on training data set.



# Use random forest because we are most concerned about accuracy and less concerned about interpretability and performance.
# train (method="rf")
#https://stackoverflow.com/questions/65282160/random-forest-cross-validated-k-fold-with-caret-package-r-best-auc

# We use parallelism because random forest is very resource intensive.
# This speeds up the training considerably.
library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

# We use a smaller number because it results in less variance
# and our sample size are fairly large so we are less concerned about bias.
train_control<- trainControl(method="cv", number=5)

rf_model <- train(classe~.,data=training, method="rf", trControl=train_control)
# look at the model results
rf_model$finalModel

# Call:
#   randomForest(x = x, y = y, mtry = param$mtry) 
# Type of random forest: classification
# Number of trees: 500
# No. of variables tried at each split: 2
# 
# OOB estimate of  error rate: 0.65%
# Confusion matrix:
#   A    B    C    D    E  class.error
# A 3828    1    0    0    1 0.0005221932
# B   14 2586    3    0    0 0.0065309259
# C    0   16 2328    3    0 0.0080954410
# D    0    0   46 2155    2 0.0217884703
# E    0    0    0    2 2468 0.0008097166

## Stop the cluster:
stopCluster(cl)

# Get out of sample error - used to estimate errors on independent data.
# Expected out of sample error is. 
rf_model_pred <- predict(rf_model,validation)
validation$predRight <- rf_model_pred == validation$classe
table(rf_model_pred,validation$classe)
# 
# rf_model_pred    A    B    C    D    E
# A 1639    8    0    0    0
# B    2 1104    9    0    0
# C    0    3  996   19    0
# D    0    0    0  924    3
# E    0    0    0    1 1055

# Therefore, the expected error rate for out of sample is:
# ( 8 + 9 + 2 + 3 + 19 + 3 + 1 ) / (1639 + 1104 + 996 + 924 + 1055) = ( 45 / 5718 ) * 100 = 0.77% 
# it is higher than in sample error rate of 0.65%

fit$mse[length(fit$mse)]


