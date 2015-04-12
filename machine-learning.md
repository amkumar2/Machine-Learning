---
title: "Machine Learning"
author: "Amit Kumar"
date: "Saturday, April 11, 2015"
output: html_document
---

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

setwd("C:/Documents and Settings/amkumar2/Training/Machine-Learning")
trainingOrg = read.csv("pml-training.csv", na.strings=c("", "NA", "NULL"))
testingOrg = read.csv("pml-testing.csv", na.strings=c("", "NA", "NULL"))
dim(trainingOrg)
dim(testingOrg)

##Pre-screening the data

###Remove variables that we believe have too many NA values.
training.dena <- trainingOrg[ , colSums(is.na(trainingOrg)) == 0]
dim(training.dena)

###Remove unrelevant variables There are some unrelevant variables that can be removed as they are unlikely to be related to dependent variable.
remove = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
training.dere <- training.dena[, -which(names(training.dena) %in% remove)]
dim(training.dere)

###Check the variables that have extremely low variance (this method is useful nearZeroVar() )
library(caret)
library(lattice)
library(ggplot2)

zeroVar= nearZeroVar(training.dere[sapply(training.dere, is.numeric)], saveMetrics = TRUE)
training.nonzerovar = training.dere[,zeroVar[, 'nzv']==0]
dim(training.nonzerovar)

###Remove highly correlated variables 90% (using for example findCorrelation() )
corrMatrix <- cor(na.omit(training.nonzerovar[sapply(training.nonzerovar, is.numeric)]))
dim(corrMatrix)

corrDF <- expand.grid(row = 1:52, col = 1:52)
corrDF$correlation <- as.vector(corrMatrix)
levelplot(correlation ~ row+ col, corrDF)

###We are going to remove those variable which have high correlation.
removecor = findCorrelation(corrMatrix, cutoff = .90, verbose = TRUE)
training.decor = training.nonzerovar[,-removecor]
dim(training.decor)

###Split data to training and testing for cross validation.

inTrain <- createDataPartition(y=training.decor$classe, p=0.7, list=FALSE)
training <- training.decor[inTrain,]; testing <- training.decor[-inTrain,]
dim(training);dim(testing)

##Analysis
###Regression Tree

###Now we fit a tree to these data, and summarize and plot it. First, we use the 'tree' package. It is much faster than 'caret' package.

library(tree)
set.seed(12345)
tree.training=tree(classe~.,data=training)
summary(tree.training)
plot(tree.training)
text(tree.training,pretty=0, cex =.8)


###tree.training

library(caret)
modFit <- train(classe ~ .,method="rpart",data=training)
print(modFit$finalModel)

##Prettier plots

library(rattle)
library(rpart.plot)
fancyRpartPlot(modFit$finalModel)

##Cross Validation

###We are going to check the performance of the tree on the testing data by cross validation.

tree.pred=predict(tree.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) 


tree.pred=predict(modFit,testing)
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) 


##Pruning tree

cv.training=cv.tree(tree.training,FUN=prune.misclass)
cv.training

plot(cv.training)

prune.training=prune.misclass(tree.training,best=18)

tree.pred=predict(prune.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) # error rate


##Random Forests
###These methods use trees as building blocks to build more complex models.


require(randomForest)

set.seed(12345)

rf.training=randomForest(classe~.,data=training,ntree=100, importance=TRUE)
rf.training
varImpPlot(rf.training,)

tree.pred=predict(rf.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) # error rate

##Conclusion
###Now we can predict the testing data from the website.

answers <- predict(rf.training, testingOrg)
answers


Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.
