require(caret) # Contains useful functions for training/test set partitioning and similar.
source("RBF.R")
library(nnet)
library(randomForest)
#data prep
set.seed(5) # For stable results. 
inTrain <- createDataPartition(y=iris$Species, p=0.75, list=FALSE) # 75% to train set
training.Iris <- iris[inTrain,]
testing.Iris <- iris[-inTrain,]

#caret
#IMPLEMENT
knn.fit <- train(Species ~ ., data = training.Iris, method = "knn")
knn.fit
predictionknn <- predict(knn.fit, testing.Iris)
confusionMatrix(predictionknn, testing.Iris$Species)

#Random Forest
print("Random Forest Results");
irisrfpred <- randomForest(Species~.,data=training.Iris,ntree=100,proximity=TRUE)
table(predict(irisrfpred),training.Iris$Species)
print(irisrfpred)

#NeuralNet
print("NeuralNet results");
irisnnpred <- nnet(Species ~ ., data=training.Iris, size=10)
nnpredicted <- predict(irisnnpred,testing.Iris,type="class")
nnclassification <- (y == apply(y,1,max)) * 1;
nnperf <- sum(abs(test.out - nnclassification)) / 2; print(nnperf) # Simple calculation of how many mis-classified
nnpred.class <- seq(1:nrow(nnclassification)); for ( i in 1:3) { nnpred.class[nnclassification[,i]==1] <- levels(iris$Species)[i]}
confusionMatrix(data = as.factor(pred.class), reference = iris[-inTrain, 5])

#INKA 
print("INKA results");
rbf <- train.inka.formula(Species~., data=training.Iris, spread=0.1, max.iter=20, classification.error.limit=0)
y <- predict.rbf(rbf, newdata=testing.Iris)
classification <- (y == apply(y, 1, max)) * 1; # INKA gives "raw" output values by default.
perf <- sum(abs(test.out - classification)) / 2; print(perf) # Simple calculation of how many mis-classified
# Confusion matrix. Requires converting one-hot to factor. 
pred.class <- seq(1:nrow(classification)); for ( i in 1:3) { pred.class[classification[,i]==1] <- levels(iris$Species)[i]}
confusionMatrix(data = as.factor(pred.class), reference = iris[-inTrain, 5])


