# inka
R implementation of Interpolating, Normalising and Kernel Allocating (INKA) neural network. 

INKA is an RBF-type neural network. Particular features of INKA are:
* The hidden layer is initially empty.
* New hidden neurons are added one at a time for the training example that has the greatest output error.
* Output layer weights are trained using a pseudo-matrix inversion.
* Outputs of hidden layer neurons are "normalized", i.e. all hidden neuron outputs are divided by their sum.
* The IMQE (Inverse Multi-Quadrics Equation) is typically used as the output function of hidden neurons, which by experience gives much better results than the Gaussian function, for instance.
* One new hidden neuron is added per iteration until the error goal has been achieved or until all training examples have been used.
* The only adjustable parameter (in practice) is the "spread" parameter that defines how wide the kernel function is. The normalisation of hidden neuron outputs makes INKA much less sensitive to the value of this parameter than what it would be otherwise. This parameter could probably be made auto-adjusted quite easily.
* Since INKA finds the "good" network size by itself and because INKA typically doesn't require hardly any parameter tuning, INKA is easy to use. 
* INKA gives the best compromise between accuracy, size and training time compared to Random Forest, Gradient Boosting, Neural Network (backpropagation) at least for task such as Iris classification, Breast Cancer diagnosis and function regression for the sombrero function, for instance.
* Work on INKA has been on stand-by 1995-2020. In practice, it is still in its early stages of development. 

# Background
INKA is an RBF-type (Radial Basis Function) neural network developed as a part of Kary Främling's PhD thesis in around 1994. It is described in the thesis and a couple of conference papers from 1995/1996.

# Running

After downloading the INKA files, make sure your R work directory is the one where you have downloaded INKA. Then execute the following for Iris classification:

```R
require(caret) # Contains useful functions for training/test set partitioning and similar.
source("RBF.R")

set.seed(2) # For stable results 
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
```

Most of the lines above are for data preparation and manipulation. The ``train``and ``predict`` lines are the only INKA-specific ones. Copy-paste of these lines into R should produce the expected result, i.e.

```
Confusion Matrix and Statistics

            Reference
Prediction   setosa versicolor virginica
  setosa         12          0         0
  versicolor      0         12         0
  virginica       0          0        12

Overall Statistics
                                     
               Accuracy : 1          
                 95% CI : (0.9026, 1)
    No Information Rate : 0.3333     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
                                     
 Mcnemar's Test P-Value : NA         

Statistics by Class:

                     Class: setosa Class: versicolor Class: virginica
Sensitivity                 1.0000            1.0000           1.0000
Specificity                 1.0000            1.0000           1.0000
Pos Pred Value              1.0000            1.0000           1.0000
Neg Pred Value              1.0000            1.0000           1.0000
Prevalence                  0.3333            0.3333           0.3333
Detection Rate              0.3333            0.3333           0.3333
Detection Prevalence        0.3333            0.3333           0.3333
Balanced Accuracy           1.0000            1.0000           1.0000
There were 50 or more warnings (use warnings() to see the first 50)
```
The warnings are related to the pseudo-matrix inversions and are not relevant.

# Notes

The examples shown here use the R ``formula`` concept in order to imitate how ``caret`` and other machine learning libraries are implemented. For more detailed information about possible parameters etc., it is recommended to look at the original, underlying lower-level functions. 
