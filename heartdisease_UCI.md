Heart disease prediction
========================

Aim of analysis
---------------

In the following document, 4 different machine learning algorithms to predict heart disease (angiographic disease status) are compared. For some algorithms, model parameters are tuned and the best model selected. The best results measured by AUC and accuracy are obtained from a logistic regression model (AUC 0.92, Accuracy 0.87), followed by Gradient Boosting Machines. From a set of 14 variables, the most important to predict heart failure are whether or not there is a reversable defect in Thalassemia followed by whether or not there is an occurrence of asymptomatic chest pain.

Dataset:
--------

Nicely prepared heart disease data are available at [UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data). The description of the database can be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names). The document mentions that previous work resulted in an accuracy of 74-77% for the preciction of heart disease using the cleveland data.

| Variable name | Short desciption                             | Variable name | Short description                                                                                 |
|---------------|----------------------------------------------|---------------|---------------------------------------------------------------------------------------------------|
| age           | Age of patient                               | thalach       | maximum heart rate achieved                                                                       |
| sex           | Sex, 1 for male                              | exang         | exercise induced angina (1 yes)                                                                   |
| cp            | chest pain                                   | oldpeak       | ST depression induc. ex.                                                                          |
| trestbps      | resting blood pressure                       | slope         | slope of peak exercise ST                                                                         |
| chol          | serum cholesterol                            | ca            | number of major vessel                                                                            |
| fbs           | fasting blood sugar larger 120mg/dl (1 true) | thal          | no explanation provided, but probably thalassemia (3 normal; 6 fixed defect; 7 reversable defect) |
| restecg       | resting electroc. result (1 anomality)       | num           | diagnosis of heart disease (angiographic disease status)                                          |

The variable we want to predict is **num** with Value 0: \< 50% diameter narrowing and Value 1: \> 50% diameter narrowing. We assume that every value with 0 means heart is okay, and 1,2,3,4 means heart disease.

From the possible values the variables can take, it is evident that the following need to be dummified because the distances in the values is random: cp,thal, restecg, slope

Data preparation
----------------

Load heart disease data and give columns names from the table above

``` r
heart.data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",header=FALSE,sep=",",na.strings = '?')
names(heart.data) <- c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                   "thalach","exang", "oldpeak","slope", "ca", "thal", "num")
```

Get a quick idea of the data

``` r
head(heart.data,3)
```

    ##   age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal
    ## 1  63   1  1      145  233   1       2     150     0     2.3     3  0    6
    ## 2  67   1  4      160  286   0       2     108     1     1.5     2  3    3
    ## 3  67   1  4      120  229   0       2     129     1     2.6     2  2    7
    ##   num
    ## 1   0
    ## 2   2
    ## 3   1

``` r
dim(heart.data)
```

    ## [1] 303  14

Explore the data quickly, how many had heart attack, women or men, age?

Values of num \> 0 are cases of heart disease. Dummify some variables.

``` r
heart.data$num[heart.data$num > 0] <- 1
barplot(table(heart.data$num),
        main="Fate", col="black")
```

![](heartdisease_UCI_files/figure-markdown_github/unnamed-chunk-3-1.png)<!-- -->

``` r
# change a few predictor variables from integer to factors (make dummies)
chclass <-c("numeric","factor","factor","numeric","numeric","factor","factor","numeric","factor","numeric","factor","factor","factor","factor")

heart.data <- convert.magic(heart.data,chclass)

heart = heart.data #add labels only for plot
levels(heart$num) = c("No disease","Disease")
levels(heart$sex) = c("female","male","")
mosaicplot(heart$sex ~ heart$num, 
           main="Fate by Gender", shade=FALSE,color=TRUE,
           xlab="Gender", ylab="Heart disease")
```

![](heartdisease_UCI_files/figure-markdown_github/unnamed-chunk-3-2.png)<!-- -->

``` r
boxplot(heart$age ~ heart$num, 
        main="Fate by Age",
         ylab="Age",xlab="Heart disease")
```

![](heartdisease_UCI_files/figure-markdown_github/unnamed-chunk-3-3.png)<!-- -->

Check for missing values - only 6 so just remove them.

``` r
s = sum(is.na(heart.data))
heart.data <- na.omit(heart.data)
#str(heart.data)
```

Training and testing data for validation
----------------------------------------

Split the data into Training (70%) and Testing (30%) data. Percentage of heart disease or not must be same in training and testing (which is handled by the R-library used here).

``` r
library(caret)
set.seed(10)
inTrainRows <- createDataPartition(heart.data$num,p=0.7,list=FALSE)
trainData <- heart.data[inTrainRows,]
testData <-  heart.data[-inTrainRows,]
nrow(trainData)/(nrow(testData)+nrow(trainData)) #checking whether really 70% -> OK
```

    ## [1] 0.7003367

Predict with 4 different methods with different tuning parameters and compare best model of each method
-------------------------------------------------------------------------------------------------------

Results are going to be stored in variable AUC. AUC is the area under the ROC which represents the proportion of positive data points that are correctly considered as positive and the proportion of negative data points that are mistakenly considered as positive. We also store Accuracy which is true positive and true negative divided by all results.

``` r
AUC = list()
Accuracy = list()
```

### Logistic regression

Only one model

``` r
set.seed(10)
logRegModel <- train(num ~ ., data=trainData, method = 'glm', family = 'binomial') 
logRegPrediction <- predict(logRegModel, testData)
logRegPredictionprob <- predict(logRegModel, testData, type='prob')[2]
logRegConfMat <- confusionMatrix(logRegPrediction, testData[,"num"])
#ROC Curve
library(pROC)
AUC$logReg <- roc(as.numeric(testData$num),as.numeric(as.matrix((logRegPredictionprob))))$auc
Accuracy$logReg <- logRegConfMat$overall['Accuracy']  #found names with str(logRegConfMat)  
```

The accuracy is 0.87 and AUC 0.92 which is already quite good.

### Random Forest model without tuning (but checked a few number of trees)

``` r
library(randomForest)
set.seed(10)
RFModel <- randomForest(num ~ .,
                    data=trainData, 
                    importance=TRUE, 
                    ntree=2000)
#varImpPlot(RFModel)
RFPrediction <- predict(RFModel, testData)
RFPredictionprob = predict(RFModel,testData,type="prob")[, 2]

RFConfMat <- confusionMatrix(RFPrediction, testData[,"num"])

AUC$RF <- roc(as.numeric(testData$num),as.numeric(as.matrix((RFPredictionprob))))$auc
Accuracy$RF <- RFConfMat$overall['Accuracy']  
```

### Boosted tree model with tuning (grid search)

Boosted tree model (gbm) with adjusting learning rate and and trees.

``` r
library(caret)
set.seed(10)
objControl <- trainControl(method='cv', number=10,  repeats = 10)
gbmGrid <-  expand.grid(interaction.depth =  c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode =10)
# run model
boostModel <- train(num ~ .,data=trainData, method='gbm',
                    trControl=objControl, tuneGrid = gbmGrid, verbose=F)
# See model output in Appendix to get an idea how it selects best model
#trellis.par.set(caretTheme())
#plot(boostModel)
boostPrediction <- predict(boostModel, testData)
boostPredictionprob <- predict(boostModel, testData, type='prob')[2]
boostConfMat <- confusionMatrix(boostPrediction, testData[,"num"])

#ROC Curve
AUC$boost <- roc(as.numeric(testData$num),as.numeric(as.matrix((boostPredictionprob))))$auc
Accuracy$boost <- boostConfMat$overall['Accuracy']  
```

### Stochastic gradient boosting

This method finds tuning parameters automatically. But a bit more work to prepare data.

``` r
# for this to work add names to all levels (numbers not allowed)
feature.names=names(heart.data)

for (f in feature.names) {
  if (class(heart.data[[f]])=="factor") {
    levels <- unique(c(heart.data[[f]]))
    heart.data[[f]] <- factor(heart.data[[f]],
                   labels=make.names(levels))
  }
}
set.seed(10)
inTrainRows <- createDataPartition(heart.data$num,p=0.7,list=FALSE)
trainData2 <- heart.data[inTrainRows,]
testData2 <-  heart.data[-inTrainRows,]


fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

set.seed(10)
gbmModel <- train(num ~ ., data = trainData2,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 tuneGrid = gbmGrid,
                 ## Specify which metric to optimize
                 metric = "ROC")
gbmPrediction <- predict(gbmModel, testData2)
gbmPredictionprob <- predict(gbmModel, testData2, type='prob')[2]
gbmConfMat <- confusionMatrix(gbmPrediction, testData2[,"num"])
#ROC Curve
AUC$gbm <- roc(as.numeric(testData2$num),as.numeric(as.matrix((gbmPredictionprob))))$auc
Accuracy$gbm <- gbmConfMat$overall['Accuracy']
```

### Support Vector Machine

``` r
set.seed(10)
svmModel <- train(num ~ ., data = trainData2,
                 method = "svmRadial",
                 trControl = fitControl,
                 preProcess = c("center", "scale"),
                 tuneLength = 8,
                 metric = "ROC")
svmPrediction <- predict(svmModel, testData2)
svmPredictionprob <- predict(svmModel, testData2, type='prob')[2]
svmConfMat <- confusionMatrix(svmPrediction, testData2[,"num"])
#ROC Curve
AUC$svm <- roc(as.numeric(testData2$num),as.numeric(as.matrix((svmPredictionprob))))$auc
Accuracy$svm <- svmConfMat$overall['Accuracy']  
```

Results: Comparison of AUC and Accuracy between models
------------------------------------------------------

``` r
row.names <- names(Accuracy)
col.names <- c("AUC", "Accuracy")
cbind(as.data.frame(matrix(c(AUC,Accuracy),nrow = 5, ncol = 2,
                           dimnames = list(row.names, col.names))))
```

    ##              AUC  Accuracy
    ## logReg 0.9161585 0.8651685
    ## RF     0.8953252 0.8089888
    ## boost  0.9095528 0.8426966
    ## gbm    0.9070122 0.8426966
    ## svm     0.882622 0.7977528

> The best model is the relative simple logistic regression model with an Area under the ROC of 0.92. We can predict heart disease with an accuracy of 0.87. The Sensitivity is 0.90 and the Specificity 0.83.

Interpretation of logistic regression model and importance of variables from boosted tree
-----------------------------------------------------------------------------------------

The coefficients of the 'best' model given AUC and Accuracy, the logistic regression model, are the following

``` r
summary(logRegModel)$coeff
```

    ##                Estimate  Std. Error    z value     Pr(>|z|)
    ## (Intercept) -2.83507855 3.547759084 -0.7991181 0.4242219263
    ## age         -0.03180882 0.029637725 -1.0732545 0.2831569344
    ## sex1         1.85291093 0.711216844  2.6052686 0.0091802253
    ## cp2          0.44982427 1.005276129  0.4474634 0.6545405141
    ## cp3         -0.51437449 0.860554826 -0.5977243 0.5500239377
    ## cp4          1.64003231 0.868294403  1.8887975 0.0589189654
    ## trestbps     0.01714664 0.015671641  1.0941190 0.2739027719
    ## chol         0.00340254 0.004918579  0.6917730 0.4890799254
    ## fbs1        -0.24155269 0.804218007 -0.3003572 0.7639046843
    ## restecg2     0.25036985 0.492666693  0.5081932 0.6113178838
    ## thalach     -0.02492582 0.014363737 -1.7353301 0.0826823533
    ## exang1       0.52595947 0.555748185  0.9463989 0.3439451656
    ## oldpeak      0.20695564 0.284315718  0.7279078 0.4666700014
    ## slope2       1.72232845 0.616791265  2.7924008 0.0052318499
    ## slope3       1.03760679 1.069301753  0.9703592 0.3318674790
    ## ca1          2.56517830 0.696934041  3.6806615 0.0002326296
    ## ca2          3.94566322 0.994034355  3.9693429 0.0000720711
    ## ca3          2.16861195 1.010144229  2.1468340 0.0318065017
    ## thal6       -0.40962979 1.003136121 -0.4083492 0.6830173544
    ## thal7        1.58273584 0.549746570  2.8790281 0.0039890275

The interpretation of the coefficient for sex, for example, is: If all predictors are held at a fixed value, the odds of getting heart disease for males (males = 1) over the odds of getting heart disease for females is exp(1.85291093) = 6.4 i.e. the odds are 540% higher.

A direct comparison of the importance of each predictor is not possible for the logistic regression model. But this could be added in further analyses - comparing predictive ability of model by removing each variable seperately. Since the boosted tree model was only slightly lower, I here show the importance of the variables calculated by the boosted tree.

``` r
boostImp =varImp(boostModel, scale = FALSE)
row = rownames(varImp(boostModel, scale = FALSE)$importance)
row = convert.names(row)
rownames(boostImp$importance)=row
plot(boostImp,main = 'Variable importance for heart failure prediction with boosted tree')
```

![](heartdisease_UCI_files/figure-markdown_github/unnamed-chunk-14-1.png)<!-- -->

Conclusion
----------

14 predictor variables from the UCI heart disease dataset are used to predict the diagnosis of heart disease (angiographic disease status). The performances of 4 different machine learning algorithms - logistic regression, boosted trees, random forest and support vector machines - are compared .

30% of the data is hold out as a testing data set that is not seen during the training stage of the data. During the training of boosted trees and support vector machines, 10-fold cross-validation is used to maximize the ROC (parameter tuning) and select the final models.

A comparison of the area under the ROC and the accuracy of the model predictions shows that logistic regression performs best (accuracy of 0.87). Tree-based methods with different tuning parameters performed slighly worse.

Nevertheless, the boosted tree model was used to compare the importance of the different variables due to the easier procedure compared to logistic regression. Having a reversable defect Thalassemia is the most important predictor in the boosted tree model, followed by asymptomatic chest pain and ST depression from exercise.

The short analysis shows the predictive capability of machine learning algorithms for heart diseases. Possible improvements can be obtained with improved data pre-processing (outliers, variances), choice of models, parameter selection, model tuning and so on.

Appendix
========

Confusion matrix output
-----------------------

Logistic Regression

``` r
logRegConfMat
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1
    ##          0 43  7
    ##          1  5 34
    ##                                           
    ##                Accuracy : 0.8652          
    ##                  95% CI : (0.7763, 0.9283)
    ##     No Information Rate : 0.5393          
    ##     P-Value [Acc > NIR] : 5.93e-11        
    ##                                           
    ##                   Kappa : 0.7277          
    ##  Mcnemar's Test P-Value : 0.7728          
    ##                                           
    ##             Sensitivity : 0.8958          
    ##             Specificity : 0.8293          
    ##          Pos Pred Value : 0.8600          
    ##          Neg Pred Value : 0.8718          
    ##              Prevalence : 0.5393          
    ##          Detection Rate : 0.4831          
    ##    Detection Prevalence : 0.5618          
    ##       Balanced Accuracy : 0.8626          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

Random Forest

``` r
RFConfMat
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1
    ##          0 39  8
    ##          1  9 33
    ##                                           
    ##                Accuracy : 0.809           
    ##                  95% CI : (0.7119, 0.8846)
    ##     No Information Rate : 0.5393          
    ##     P-Value [Acc > NIR] : 9.657e-08       
    ##                                           
    ##                   Kappa : 0.6163          
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.8125          
    ##             Specificity : 0.8049          
    ##          Pos Pred Value : 0.8298          
    ##          Neg Pred Value : 0.7857          
    ##              Prevalence : 0.5393          
    ##          Detection Rate : 0.4382          
    ##    Detection Prevalence : 0.5281          
    ##       Balanced Accuracy : 0.8087          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

Boosted tree

``` r
boostConfMat
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1
    ##          0 41  7
    ##          1  7 34
    ##                                           
    ##                Accuracy : 0.8427          
    ##                  95% CI : (0.7502, 0.9112)
    ##     No Information Rate : 0.5393          
    ##     P-Value [Acc > NIR] : 1.452e-09       
    ##                                           
    ##                   Kappa : 0.6834          
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.8542          
    ##             Specificity : 0.8293          
    ##          Pos Pred Value : 0.8542          
    ##          Neg Pred Value : 0.8293          
    ##              Prevalence : 0.5393          
    ##          Detection Rate : 0.4607          
    ##    Detection Prevalence : 0.5393          
    ##       Balanced Accuracy : 0.8417          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

Gradient boosting

``` r
gbmConfMat
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction X1 X2
    ##         X1 42  8
    ##         X2  6 33
    ##                                           
    ##                Accuracy : 0.8427          
    ##                  95% CI : (0.7502, 0.9112)
    ##     No Information Rate : 0.5393          
    ##     P-Value [Acc > NIR] : 1.452e-09       
    ##                                           
    ##                   Kappa : 0.6823          
    ##  Mcnemar's Test P-Value : 0.7893          
    ##                                           
    ##             Sensitivity : 0.8750          
    ##             Specificity : 0.8049          
    ##          Pos Pred Value : 0.8400          
    ##          Neg Pred Value : 0.8462          
    ##              Prevalence : 0.5393          
    ##          Detection Rate : 0.4719          
    ##    Detection Prevalence : 0.5618          
    ##       Balanced Accuracy : 0.8399          
    ##                                           
    ##        'Positive' Class : X1              
    ## 

Support vector machine

``` r
svmConfMat
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction X1 X2
    ##         X1 37  7
    ##         X2 11 34
    ##                                           
    ##                Accuracy : 0.7978          
    ##                  95% CI : (0.6993, 0.8755)
    ##     No Information Rate : 0.5393          
    ##     P-Value [Acc > NIR] : 3.388e-07       
    ##                                           
    ##                   Kappa : 0.5959          
    ##  Mcnemar's Test P-Value : 0.4795          
    ##                                           
    ##             Sensitivity : 0.7708          
    ##             Specificity : 0.8293          
    ##          Pos Pred Value : 0.8409          
    ##          Neg Pred Value : 0.7556          
    ##              Prevalence : 0.5393          
    ##          Detection Rate : 0.4157          
    ##    Detection Prevalence : 0.4944          
    ##       Balanced Accuracy : 0.8001          
    ##                                           
    ##        'Positive' Class : X1              
    ## 

Example of Model output for selection of tuning parameters
----------------------------------------------------------

``` r
boostModel
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 208 samples
    ##  13 predictor
    ##   2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 188, 188, 186, 187, 188, 186, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa    
    ##   1                    50     0.8269697  0.6494118
    ##   1                   100     0.8126840  0.6212485
    ##   1                   150     0.8276623  0.6515717
    ##   1                   200     0.8088312  0.6126281
    ##   1                   250     0.8185714  0.6311624
    ##   1                   300     0.7985714  0.5904090
    ##   1                   350     0.8083333  0.6104113
    ##   1                   400     0.7885931  0.5718116
    ##   1                   450     0.7883550  0.5718138
    ##   1                   500     0.7881169  0.5708208
    ##   1                   550     0.7883550  0.5712016
    ##   1                   600     0.7792641  0.5517715
    ##   1                   650     0.7785931  0.5522618
    ##   1                   700     0.7835931  0.5620536
    ##   1                   750     0.7883550  0.5713787
    ##   1                   800     0.7885714  0.5709164
    ##   1                   850     0.7883550  0.5716414
    ##   1                   900     0.7838095  0.5620844
    ##   1                   950     0.7838095  0.5620844
    ##   1                  1000     0.7838095  0.5620844
    ##   1                  1050     0.7885714  0.5719349
    ##   1                  1100     0.7885714  0.5719349
    ##   1                  1150     0.7838095  0.5623471
    ##   1                  1200     0.7790260  0.5533788
    ##   1                  1250     0.7835714  0.5621431
    ##   1                  1300     0.7788095  0.5522926
    ##   1                  1350     0.7792641  0.5534057
    ##   1                  1400     0.7745022  0.5422693
    ##   1                  1450     0.7745022  0.5437322
    ##   1                  1500     0.7695022  0.5329953
    ##   5                    50     0.8226623  0.6406846
    ##   5                   100     0.8176623  0.6297134
    ...   .                  ...      ........   ........

    ##   9                  1050     0.7652381  0.5227102
    ##   9                  1100     0.7652381  0.5227102
    ##   9                  1150     0.7750000  0.5417366
    ##   9                  1200     0.7750000  0.5417366
    ##   9                  1250     0.7750000  0.5417366
    ##   9                  1300     0.7750000  0.5417366
    ##   9                  1350     0.7795455  0.5516164
    ##   9                  1400     0.7797835  0.5529154
    ##   9                  1450     0.7847835  0.5635403
    ##   9                  1500     0.7847835  0.5635403
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final values used for the model were n.trees = 150,
    ##  interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.
