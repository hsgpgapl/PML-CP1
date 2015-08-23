# Practical Machine Learning - Class Project
Mark Friedman  
Thursday, August 06, 2015  
## Summary

Create a prediction model using personal activity data from Groupware@LES (http://groupware.les.inf.puc-rio.br/har).  The goal is to predict which exercise method was used by the respondent.  The details of the dataset are provided on the website, but in brief are as follows:  six test subjects were asked to perform ten repetitions of a physical activity in five different ways.  The goal of this paper is to use the machine learning course material to test how well a model can be built that predicts which way the exercise was performed, based on the collected data. 

## Requirements
1. Describe how the model was built  
2. Describe how cross validation was used  
3. Identify the expected out of sample error rate  
4. Explain the choices made




```r
library(lattice)
library(ggplot2)
library(caret)
#library(MASS)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
#library(rpart)
set.seed(123456)
```

### Data Preparation
Based on course materials, the first step was to remove variables in order to reduce the amount of data needed, and improve the ability to create a model.  These data preparation steps included creating removing variables that had low variance, had missing values, and finally removing variables that were not useful in a model (e.g., name of subject).


```r
# Read in Downloaded Data and set missing values

training <- read.csv("pml-training.csv", header = TRUE, na.strings=c("NA","#DIV/0!"))
testing  <- read.csv("pml-testing.csv", header = TRUE, na.strings=c("NA","#DIV/0!"))

# Clean/pre-process and keep variables useful for modeling.

# find columns wth missing values
drop.columns1 <- names(which(colSums(is.na(training))>0))
training <- training[,!(names(training) %in% drop.columns1)]
testing  <- testing[,!(names(testing) %in% drop.columns1)]
#remove(drop.columns1)

# find columns with low variance
drop.columns2 <- nearZeroVar(training, freqCut = 95/5, uniqueCut = 10, saveMetrics = FALSE, foreach = FALSE, allowParallel = TRUE)
drop.columns.names2 <- names(training[drop.columns2])
training <- training[,!(names(training) %in% drop.columns.names2)]
testing  <- testing[,!(names(testing) %in% drop.columns.names2)]
#remove(drop.columns2)

# drop unnecessary columns
drop.columns.names3 <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window")
training <- training[,!(names(training) %in% drop.columns.names3)]
testing <- testing[,!(names(testing) %in% drop.columns.names3)]
#remove(drop.columns.names3)
```
### Create Validation Dataset
We take the training dataset and further split intoa validation dataset to see how well the model works


```r
# Split the training to create cross-validation file
inCrossVal <- createDataPartition(training$classe, p=0.6, list=FALSE)
training.data <- training[-inCrossVal,]
crossval.data <- training[inCrossVal,]
#remove(inCrossVal)
```

### Create Various Models
The author admits he was dumbfounded doing this project, and thus did several models in an attempt to see if

* 1 They ran, and 
* 2 They produced output  

in the end, he picked the one (Random Forest) that seemed to produce usable output, and commented out the rest.


```r
modelFit <- train(classe ~ ., data=training.data, preProcess=c("pca"), trControl=trainControl(method="cv"), method="rf")

modelFit
```

```
## Random Forest 
## 
## 7846 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction, scaled, centered 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 7061, 7062, 7061, 7061, 7061, 7061, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
##    2    0.9474891  0.9335688  0.008010406  0.01013832
##   27    0.9365303  0.9196991  0.009944688  0.01259505
##   52    0.9352567  0.9180838  0.010705786  0.01355508
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
modelFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 5.02%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 2180   16   15   16    5  0.02329749
## B   43 1414   46    5   10  0.06851120
## C    7   27 1305   19   10  0.04605263
## D    9    9   79 1186    3  0.07776050
## E    1   22   33   19 1367  0.05201110
```

```r
#tsub.lda <- train(classe~., data=trainingsub, method="lda")
#tsub.lda

#tsub.rf <- randomForest(classe ~. , data=trainingsub, method="class")
#tsub.rf

#tsub.rf2 <- train(classe ~. , preProcess=c("BoxCox"), data=trainingsub, method="rf")
#tsub.rf2

#tsub.rf3 <- train(classe ~. , preProcess=c("pca"), trControl=trainControl(method="cv"), data=trainingsub, method="rf")
#tsub.rf3

#csub.lda <- train(classe~., data=crossvalsub, method="lda")
#csub.lda
#csub.rf <- randomForest(classe ~. , data=crossvalsub, method="class")
#csub.rf

#predict.rpart <- train(classe~. , data = training, method="rpart")
#ctrl.rf <- trainControl(allowParallel=T, method="cv", number=4)
#predict.rf <- train(classe ~ ., data=training, model="rf", trControl=ctrl.rf)
```

### Check Using Cross Validation
The model was then applied to the cross validation dataset.


```r
classe.index <-which(colnames(crossval.data)=="classe")
predict.cv <- predict(modelFit, crossval.data[,-classe.index])
```
### Out of Sample Error Rate
The out of sample error rate is calculated by taking the model from the training data, and applying to the testing data.  I expect the accuracy to be better than or roughly equal to the training data.  This demonstrates that the model is not overfitting.  The traing error was 0.94 and the testing was 0.94.


```r
confusionMatrix(predict.cv, crossval.data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3284   81   13   13    3
##          B   32 2111   84    7   28
##          C    9   71 1930  147   28
##          D   17    6   24 1750   26
##          E    6   10    3   13 2080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9473          
##                  95% CI : (0.9431, 0.9512)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9333          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9809   0.9263   0.9396   0.9067   0.9607
## Specificity            0.9869   0.9841   0.9738   0.9926   0.9967
## Pos Pred Value         0.9676   0.9332   0.8833   0.9600   0.9848
## Neg Pred Value         0.9924   0.9823   0.9871   0.9819   0.9912
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2789   0.1793   0.1639   0.1486   0.1766
## Detection Prevalence   0.2882   0.1921   0.1855   0.1548   0.1793
## Balanced Accuracy      0.9839   0.9552   0.9567   0.9497   0.9787
```

### Explain Choices Made
I selected Random Forest because I was able to get it to work..


##Acknowledgements
Estimating accuracy from //http://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/. 

Data are graciously provided by Groupware@les
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more at: http://groupware.les.inf.puc-rio.br/har#ixzz3jDWEZJOH

##Libraries and Environment

The following libraries are needed for reproducing the output: *randomForest, caret, ggplot2, lattice, stats, graphics, grDevices, utils, datasets, methods, base*

This run was created with R version 3.2.1 (2015-06-18) on a windows x64 bit machine.
