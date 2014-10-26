# Practical Machine Learning Assignment
Erik Larson  
October 25, 2014  
##Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, I use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

The goal of the project is to be able to use this data to predict the manner in which they did the exercise. This is the "classe" variable in the training set

I will use a random forest algorithm and 4 fold cross validation

##Loading Data
First I load the data from the csv files provided as part of the assignment.


```r
pml_training<-read.csv('pml-training.csv')
pml_testing<-read.csv('pml-testing.csv')
```

##Calculating Sample Error

A random forrest model was selected because it predicts well with non-linear data and tends to not overfit.

Looking through the dataset, there are many columns with missing data or NAs.  I first remove columns with missing data.  

Then I loop 4 times over the data, selecting training and cross validation sets based on a 4 fold cross validation.  After the model is trained, I compare the predictions on the cross validation set against the actual data.  I take the number of matches divided by the total observations as the accuracy of the model.

I then average the accuracy calcs together to predict the out of sample accuracy for the model.


```r
library(caret)
library(randomForest)
library('scales')

set.seed(1223)

NUM_FOLDS<-4
folds <- createFolds(pml_training$X,k=NUM_FOLDS)

#Take out columns with empty values
bad_columns<-c(1,12:36,50:59,69:83,87:101,103:112,125:151)
models <- vector("list",NUM_FOLDS)
accuracy <- c()

#Set the seed so that results are consistant

for(i in 1:NUM_FOLDS){
        train <- pml_training[-folds[[i]],-bad_columns]
        train$cvtd_timestamp=as.POSIXct(train$cvtd_timestamp,format="%d/%m/%Y %H:%M")
        cv_test <- pml_training[folds[[i]],-bad_columns]
        cv_test$cvtd_timestamp=as.POSIXct(cv_test$cvtd_timestamp,format="%d/%m/%Y %H:%M")
        
        models[[i]] <- randomForest( classe ~ ., data=train, verbose = TRUE,ntree=310)
        predictions <- predict(models[[i]],cv_test)
        accuracy[[i]]<-sum(predictions==cv_test$classe)/(length(predictions)*1.0)
}
mean_oos_accuracy<-mean(accuracy)
```


```r
library('scales')
formatted_oss_accuracy<-percent(mean_oos_accuracy)
```
The average out of sample accuracy is predicted to be 99.9%.  That means that all of the models are equally good for predicting classes.

Below I plot the in-sample error rate against the number of trees used for the random forest algorithm. The error approaches zero after around 20 trees are used.  With 300 trees, there is close to 0 error.


```r
library(caret)
plot(models[[1]])
```

![](./ml_assignment_files/figure-html/plotmodel-1.png) 

I will use the first model to predict values provided from pml_testing dataset.  I need to adjust the new_window variable so that it matches the format of the training set.  You can see the predicted results below.  These got me 100% accuracy on the submitted results for the class.


```r
library(caret)
model<-models[[1]]

test <- pml_testing[,-bad_columns]
test$cvtd_timestamp=as.POSIXct(test$cvtd_timestamp,format="%d/%m/%Y %H:%M")
levels(test$new_window)=c('no','yes')

predictions <- predict(model,test)
answers <- predictions

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
print(answers)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


