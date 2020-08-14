require(caret) # Contains useful functions for training/test set partitioning and similar.
source("RBF.R")

set.seed(5) # For stable results. 
inTrain <- createDataPartition(y=iris$Species, p=0.75, list=FALSE) # 75% to train set
training.Iris <- iris[inTrain,]
testing.Iris <- iris[-inTrain,]
rbf <- train.inka.formula(Species~., data=training.Iris, spread=0.1, max.iter=20, classification.error.limit=0)
y <- predict.rbf(rbf, newdata=testing.Iris)
classification <- (y == apply(y, 1, max)) * 1; # INKA gives "raw" output values by default.
perf <- sum(abs(test.out - classification)) / 2; print(perf) # Simple calculation of how many mis-classified

# Confusion matrix. Requires converting one-hot to factor. 
pred.class <- seq(1:nrow(classification)); for ( i in 1:3) { pred.class[classification[,i]==1] <- levels(iris$Species)[i]}
confusionMatrix(data = as.factor(pred.class), reference = iris[-inTrain, 5])

