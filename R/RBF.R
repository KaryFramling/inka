# "R" implementation of RBF Neural Network, done in OOP fashion.
#
# Kary Främling, created 28 dec 2005
#


#' Create new RBF network object.
#'
#' @param nbrInputs blabla
#' @param nbrOutputs blabla
#' @param n.hidden.neurons blabla
#' @param activation.function blabla
#' @param output.function blabla
#' @param normalize blabla
#' @param spread blabla
#' @return New RBF network (class RBF, inherits FunctionApproximator),
#' with following properties:
#' - nbrInputs: Number of inputs
#' - nbrOutputs: Number of outputs
#' - n.hidden.neurons: Number of hidden neurons (allowed to provide zero as value)
#' - activation.function: The activation function to be used by hidden layer neurons
#' - output.function: Output function to use for hidden layer neurons
#' - normalize: Normalize hidden layer outputs or not, i.e. "ordinary" RBF or normalized RBF
#' @export
#' @examples
#' # Create RBF network manually, one input, one output. "Normalised" RBF (default)
#' # First empty hidden layer, then configure weights manually.
#' rbf <- rbf.new(1, 1, 0)
#' ol <- rbf$get.outlayer()
#' ol$set.weights(matrix(c(1,2),nrow=1,ncol=2))
#' hl <- rbf$get.hidden()
#' hl$set.weights(matrix(c(1,5),nrow=2,ncol=1,byrow=TRUE))
#'
#' # See what it looks like. Centroids at x=1 and x=5. Slight overshooting
#' # in y-values, which is normal (no training yet).
#' x <- matrix(seq(-5,10,0.1))
#' y <- rbf$eval(x)
#' plot(x, y, type='l')
#'
#' # Example of matrix inversion "learning" on output layer weights.
#' # Create small training set and train with QR-decomposition
#' x <- matrix(c(0,0,1,1,1,0,0,1), ncol=2, byrow=TRUE)
#' y <- matrix(c(0,1,0.3,0.6),ncol=1)
#' n.hidden <- 4
#' rbf <- rbf.new(2, 1, n.hidden)
#' hl <- rbf.classifier.new(nbrInputs=2, nbrOutputs=n.hidden)
#' rbf$set.hidden(hl)
#' mins <- c(0,0); maxs <- c(1,1)
#' xints <- yints <- 2
#' hl$init.centroids.grid(mins, maxs, c(xints,yints))
#' rbf$set.spread(0.5)
#' outp <- rbf$eval(x)
#' h.out <- hl$get.outputs()
#' w <- qr.solve(h.out, y)
#' ol <- rbf$get.outlayer()
#' ol$set.weights(t(w))
#'
#' # See if the examples were learned correctly.
#' rbf$eval(x)
#'
#' # Plot function in 3D
#' xp <- yp <- seq(0,1,0.05)
#' m <- expand.grid(xp,yp)
#' z <- rbf$eval(m)
#' zm <- matrix(data=z, nrow=length(xp), ncol=length(yp), byrow=TRUE)
#' persp(xp, yp, zm, zlim=c(-0.1,1.1), theta=15, phi=20, shade=0.3, ticktype="detailed")
rbf.new <- function(nbrInputs, nbrOutputs, n.hidden.neurons,
                    activation.function=squared.distance.activation,
                    output.function=imqe.output.function,
                    normalize=TRUE, spread=0.1) {

  ninputs <- nbrInputs
  noutputs <- nbrOutputs
  nhidden <- n.hidden.neurons
  outputs <- c()
  inputs <- c()
  targets <- c()
  formula <- NULL

  # Create neural layers
  outlayer <- adaline.new(nhidden, noutputs)
  hidden <- rbf.classifier.new(ninputs, nhidden, activation.function, output.function)
  hidden$set.normalize(normalize)
  if ( normalize ) { # Remove bias term
    outlayer$set.use.bias(FALSE)
  }

  # Set default parameters
  hidden$set.spread(spread)

  # Method for doing forward-pass, i.e. calculate outputs based on inputs
  eval <- function(invals) {
    # # Deal with data.frame and similar cases. This could maybe be done more elegantly somehow...
    # if ( !is.matrix(invals) )
    #   invals <- as.matrix(invals)
    inputs <<- invals
    h <- hidden$eval(inputs)
    outputs <<- outlayer$eval(h)
    return(outputs)
  }

  # There are many possible ways of training. This one is for
  # Widrow-Hoff on output layer. An "eval" must
  # always be performed prior to calling "train" in order to set input
  # values and evaluated values correctly.
  # The number of rows in "input" matrix has to be the same as in the
  # "target" matrix.
  train <- function(t) {
    targets <<- t
    outlayer$train(t)
  }

  # Return RBF object with data only.
  #
  # Create and return list object with class "inka", that contains all weights,
  # parameters etc of this RBF object.
  #
  # @return `inka` object.
  as.inka <- function() {
    weights <- list()
    weights[[1]] <- hidden$get.weights()
    weights[[2]] <- outlayer$get.weights()
    inka <- list(
      weights = weights,
      # activation.functions = activation.functions,
      # output.functions = output.functions,
      spread = hidden$get.spread(),
      normalize = hidden$get.normalize(),
      formula = formula
    )
    class(inka) <- c("inka", class(inka))
    return(inka)
  }

  # Return list of "public" methods
  pub <- list(
    get.inputs = function() { inputs },
    get.nbr.inputs = function() { ninputs },
    get.outputs = function() { outputs },
    get.nbr.outputs = function() { noutputs },
    get.outlayer = function() { outlayer },
    get.hidden = function() { hidden },
    get.spread = function() { hidden$get.spread() },
    get.nrbf = function() { hidden$get.normalize() },
    set.outlayer = function(o) { outlayer <<- o },
    set.hidden = function(o) { hidden <<- o },
    set.spread = function(value) { hidden$set.spread(value) },
    set.nrbf = function(value) { hidden$set.normalize(value) },
    eval = function(invals) { eval(invals) },
    train = function(t) { train(t) },
    get.formula = function() { formula },
    set.formula = function(f) { formula <<- f },
    as.inka = function() { as.inka() }
  )

  # We implement "FunctionApproximator"
  fa <- function.approximator.new()

  class(pub) <- c("RBF",class(fa),class(pub))
  return(pub)

}

#' INKA training algorithm, as in my (Kary's) PhD thesis.
#'
#' @param rbf RBF network instance to use
#' @param train.inputs Matrix of input values of the training set
#' @param train.outputs Matrix of target output values of the training set
#' @param c "c" paremeter in INKA, i.e. minimal allowed distance between neurons
#' @param max.iter Maximal number of training iterations
#' @param inv.whole.set.at.end If TRUE, then pseudo-matrix is calculated for
#' entire traning set at the end.
#' @param classification.error.limit If specified, training ends when the
#' number of classification errors goes under this value. To be used for
#' classification tasks.
#' @param rmse.limit If specified, traning ends when RMSE between target
#' values and output values goes under (or equals) the given limit.
#' @param test.inputs blabla
#' @param test.outputs blabla
#'
#' @export
train.inka <- function(rbf, train.inputs, train.outputs, c=0.01, max.iter=1,
                       inv.whole.set.at.end=F, classification.error.limit=NULL, rmse.limit=NULL,
                       test.inputs=NULL, test.outputs=NULL) {

  # Sanity checks
  if ( is.null(classification.error.limit) && is.null(rmse.limit) )
    stop("Both \"classification.error.limit\" and \"rmse.limit\" are NULL in train.inka!")

  # Get number of inputs, number of outputs.
  ninps <- rbf$get.nbr.inputs()
  noutps <- rbf$get.nbr.outputs()

  # Old remains: turn everything into matrix, just in case.
  train.inputs <- as.matrix(train.inputs)
  train.outputs <- as.matrix(train.outputs)

  # Initialize set of remaining (not yet used for creating hidden neuron) traning examples.
  # Set of used examples is initially empty.
  t.in.remain <- matrix(train.inputs, ncol=ninps) # The set of remaining/unused training examples.
  targets.remain <- matrix(train.outputs, ncol=noutps) # matrix is to deal with case of it being a vector
  t.in.used <- NULL
  targets.used <- NULL

  # Loop until stopping condition, e.g. on some error measure, number of hodden neurons or whatever.
  for ( iter.ctr in 1:max.iter ) {
    ty <- rbf$eval(t.in.remain) # Do forward pass for all "remaining" training examples

    # Find out which one has the greatest error, taken over all outputs
    diff <- abs(targets.remain - ty) # Estimation error for whole training set, absolute value
    rdiff <- apply(diff, 1, max) # We only want to know which rows have greatest error
    max.error <- max(rdiff) # Largest error
    indices <- which(rdiff %in% max.error) # Indices of rows with largest error
    n.max <- length(indices)
    if ( n.max > 1 ) { # If more than one row with largest error, then take a random one of them.
      i <- 1 + round(runif(1)*n.max)
      if ( i > n.max )
        i <- n.max # Just for unprobable case of runif = 1
      ind.larg.err <- indices[i]
    } else {
      ind.larg.err <- indices[1]
    }

    # Add new hidden neuron
    new.centroid <- t.in.remain[ind.larg.err,]
    new.tout <- targets.remain[ind.larg.err,]

    # Create set of "used" examples if this is the first one, otherwise append new example to it.
    if ( is.null(t.in.used) ) {
      t.in.used <- matrix(new.centroid, nrow=1)
      targets.used <- matrix(new.tout, nrow=1)
    } else {
      t.in.used <- rbind(t.in.used, new.centroid)
      targets.used <- rbind(targets.used, new.tout)
    }

    # Remove from set of remaining/unused training examples.
    t.in.remain <- matrix(t.in.remain[-ind.larg.err,], ncol=ninps)
    targets.remain <- matrix(targets.remain[-ind.larg.err,], ncol=noutps)

    # Add new hidden neuron and output weight. But only if far enough from existing centroids
    # TO BE IMPLEMENTED!
    # if ( new.centroid far enough from existing ones ) {}
    # Get distance of hidden neuron centroids to all data points.
    #    hw <- hl$get.weights()
    #    ref <- hw[1,]
    #    t.aff <- t(aff.trans$eval(t(t.in)))
    #    apply(t.aff,1,function(x)sqrt(sum((x-ref)^2)))

    # Add one more column to outpur layer weights, initialize to zero.
    ol <- rbf$get.outlayer()
    ow <- ol$get.weights()
    ow <- cbind(ow, 0)
    ol$set.weights(ow)

    # Add one more neuron (row) to hidden layer with centroid initialized to training example.
    hl <- rbf$get.hidden()
    hw <- hl$get.weights()
    hw <- rbind(hw, new.centroid)
    hl$set.weights(hw)

    # Do forward pass of RBF net for re-calculating hidden layer outputs.
    ty <- rbf$eval(t.in.used)
    h.out <- hl$get.outputs()

    # Get pseudo-inverse matrix between hidden layer outputs and the correct/target values
    # and use that as new weights for output layer.
    w <- qr.solve(h.out, targets.used)
    ol$set.weights(t(w))

    # Check if error goal has been achieved.
    if (is.null(test.inputs)) inps <- train.inputs else inps <- test.inputs
    if (is.null(test.outputs)) outps <- train.outputs else outps <- test.outputs
    y <- rbf$eval(inps)
    if ( !is.null(classification.error.limit) ) {
      if ( ncol(outps) == 1) {perf <- sum(abs(outps - round(y)))} # Should maybe restrict to 0,1?
      else {
        classification <- (y == apply(y, 1, max)) * 1
        perf <- sum(abs(outps - classification)) / 2 # One error gives sum on 2.
      }
      if ( perf <= classification.error.limit ) # Stop training if error is small enough.
        break
    }
    else {
      # Check if RMSE goal has been achieved, on training set for the moment.
      if ( !is.null(rmse.limit) ) {
        perf <- root.mean.squared.error(outps, y)
        if ( perf <= rmse.limit ) break # Stop training if error is small enough.
      }
    }
  }

  # Do matrix inversion for whole training set at the end
  if ( inv.whole.set.at.end ) {
    ty <- rbf$eval(train.inputs)
    h.out <- hl$get.outputs()
    w <- qr.solve(h.out, train.outputs)
    ol$set.weights(t(w))
  }
  return(nrow(hl$get.weights())) # Return number of hidden neurons created
}

#' Create "n" RBF nets with given parameters and return the one that
#' performs the best.
#'
#' @param n blabla
#' @param train.inputs blabla
#' @param train.outputs blabla
#' @param max.iter blabla
#' @param inv.whole.set.at.end blabla
#' @param classification.error.limit Limit of misclassified instances to reach
#' before ending the training. Either "classification.error.limit" or
#' "rmse.limit" MUST be provided!
#' @param rmse.limit RMSE limit to reach before ending the training.
#' Either "classification.error.limit" or "rmse.limit" MUST be provided!
#' @param activation.function blabla
#' @param output.function blabla
#' @param nrbf blabla
#' @param use.bias blabla
#' @param spread blabla
#' @param c blabla
#' @param test.inputs blabla
#' @param test.outputs blabla
#'
#' @export
find.best.inka <- function(n=1, train.inputs, train.outputs, max.iter=1,
                           inv.whole.set.at.end=F, classification.error.limit=NULL,
                           rmse.limit=NULL, activation.function=squared.distance.activation,
                           output.function=imqe.output.function, nrbf=T, use.bias=F,
                           spread=0.1, c=0.01, test.inputs=NULL, test.outputs=NULL) {
  # Get number of inputs and number of outputs. As.matrix is to deal with the case if
  # either one is a vector only.
  n.in <- ncol(as.matrix(train.inputs))
  n.out <- ncol(as.matrix(train.outputs))

  # Iterate until best one os found
  best.rbf <- NULL
  best.perf <- NULL
  best.n.hidden <- NULL
  for ( i in seq(1:n) ) {
    # Create new RBF
    rbf <- rbf.new(n.in, n.out, 0, activation.function, output.function, normalize=nrbf, spread=spread)

    # Train
    n.hidden <- train.inka(rbf, train.inputs, train.outputs, c, max.iter,
                           inv.whole.set.at.end, classification.error.limit,
                           rmse.limit, test.inputs, test.outputs)
    # If better than previous, replace best.rbf
    # Check if classification error goal has been achieved.
    if (is.null(test.inputs)) inps <- train.inputs else inps <- test.inputs
    if (is.null(test.outputs)) outps <- train.outputs else outps <- test.outputs
    y <- rbf$eval(inps)
    if ( !is.null(classification.error.limit) ) {
      if ( ncol(outps) == 1) {perf <- sum(abs(outps - round(y)))} # Should maybe restrict to 0,1?
      else {
        classification <- (y == apply(y, 1, max)) * 1
        perf <- sum(abs(outps - classification)) / 2 # One error gives sum on 2.
      }
    }
    else {
      # Check if RMSE goal has been achieved, on training set for the moment.
      if ( !is.null(rmse.limit) ) {perf <- root.mean.squared.error(outps, y)}
    }

    # See if improved
    if ( is.null(best.rbf) || perf <= best.perf ) {
      if ( is.null(best.n.hidden) || n.hidden < best.n.hidden ) { # We prefer smaller nets
        best.rbf <- rbf
        best.perf <- perf
        best.n.hidden <- n.hidden
      }
    }
  }
  return (best.rbf)
}

# Extract all elements from formula in form expected by INKA implementation.
# If output variable is a factor, then split it into corresponding number of one-hot outputs.
# Return a list of elements:
# 1. Name of output variable
# 2. Matrix of input values
# 3. Matrix of output/target values.
#' @import stats
parse.formula.inka <- function(formula, data) {
  out.name <- formula[[2]] # We expect that output variable name is given
  if ( sum(names(data)==out.name) > 0 ) {
    d <- model.frame(formula=formula, data=data) # Do any default treatment on data first
    outputs <- d[, names(d)==out.name, drop = FALSE] # Extract output matrix
    if ( is.factor(outputs[,1]) ) { # One-hot if factor
      n <- levels(outputs[,1])
      outputs <- model.matrix(~0+outputs[,1])
      attr(outputs, "dimnames")[[2]] <- n
    }
  }
  else {
    d <- data
    outputs <- NULL
  }
  inputs <- (d[, names(d)!=out.name]) # Extract input matrix. We suppose there always has to be at least one input.
  inputs <- apply(inputs,2,as.numeric) # Just in case values became strings...
  return(list(out.name=out.name, inputs=inputs, outputs=outputs))
}

#' Train with formula / data call
#'
#' If dependent variable is of type "factor", then it is automatically one-hot encoded.
#' @param formula blabla
#' @param data blabla
#' @param spread blabla
#' @param normalize blabla
#' @param ... blabla
#'
#' @export
train.inka.formula <- function(formula, data, spread=0.1, normalize=TRUE, ...) {
  # Split into parameters as expected by original INKA methods.
  pars <- parse.formula.inka(formula, data)
  out.name <- pars$out.name; inputs <- pars$inputs; outputs <- pars$outputs
  rbf <- rbf.new(ncol(inputs), ncol(outputs), 0, normalize=normalize, spread=spread)
  rbf$set.formula(formula)
  train.inka(rbf=rbf, train.inputs=inputs, train.outputs=outputs, ...)
  return(rbf)
}

#' Train with formula / data call
#'
#' If dependent variable is of type "factor", then it is automatically
#' one-hot encoded.
#' @param formula blabla
#' @param data blabla
#' @param ... blabla
#'
#' @export
find.best.inka.formula <- function(formula, data, ...) {
  # Split into parameters as expected by original INKA methods.
  pars <- parse.formula.inka(formula, data)
  out.name <- pars$out.name; inputs <- pars$inputs; outputs <- pars$outputs
  rbf <- find.best.inka(train.inputs=inputs, train.outputs=outputs, ...)
  rbf$set.formula(formula)
  return(rbf)
}

#' Caret-like "predict" method with formula.
#'
#' However, here the default is to return the real-valued output values
#' also for classification tasks, rather than doing one-hot coding.
#' @param object blabla
#' @param newdata blabla
#' @param ... Same as for generic "predict".
#' @export
predict.RBF <- function(object, newdata, ...) {
  if ( !inherits(object, "RBF") )
    stop("predict.RBF called with \"object\" that is not of class RBF")
  f <- object$get.formula()
  if ( is.null(f) )
    stop("Formula has to be given before calling predict.RBF.")
  pars <- parse.formula.inka(formula=f, data=newdata)
  inputs <- pars$inputs
  return(object$eval(inputs))
}

#=========================================================================
# After this comes development-time code, for testing etc.
#=========================================================================

# Matrix inversion for learning output layer weights
# y = Ax (slight confusion with variable naming in R documentation in my opinion)
# Two inputs, four data points (overdetermined). With Bias term.
# x <- matrix(c(
#   0,0,1,
#   5,5,1,
#   5,0,1,
#   0,5,1
# ), nrow=4, byrow=T)
# y <- matrix(c(0.1,1,0.55,0.55))
# H <- qr.solve(x, y)
# x%*%H

# Create small training set and train with QR-decomposition
# rbf.regression.test <- function(indices=c(1)) {
#   x <- matrix(c(
#     0,0,
#     1,1,
#     1,0,
#     0,1
#   ), ncol=2, byrow=T)
#   y <- matrix(c(0,1,0.3,0.6),ncol=1)
#   xints <- 2
#   yints <- 2
#   n.hidden <- xints*yints
#   rbf <- rbf.new(2, 1, n.hidden, activation.function=squared.distance.activation,
#                  output.function=imqe.output.function)
#   hl <- rbf.classifier.new(nbrInputs=2, nbrOutputs=n.hidden, activation.function=squared.distance.activation,
#                            output.function=gaussian.output.function)
#   mins <- c(0,0)
#   maxs <- c(1,1)
#   #at <- scale.translate.ranges(mins, maxs, c(0,0), c(1,1))
#   #hl$init.centroids.grid(mins, maxs, c(xints,yints),affine.transformation=at)
#   hl$init.centroids.grid(mins, maxs, c(xints,yints))
#   rbf$set.hidden(hl)
#   rbf$set.nrbf(TRUE)
#   rbf$set.spread(0.3)
#   outp <- rbf$eval(x)
#   h.out <- hl$get.outputs()
#   w <- qr.solve(h.out, y)
#   ol <- rbf$get.outlayer()
#   ol$set.weights(t(w))
#   xp <- seq(0,1,0.05)
#   yp <- xp
#   m<-create.permutation.matrix(list(xp,yp))
#   z <- rbf$eval(m)
#   zm <- matrix(data=z, nrow=length(xp), ncol=length(yp), byrow=TRUE)
#   persp(xp, yp, zm, zlim=c(-0.1,1.1), theta=15, phi=20, shade=0.3, ticktype="detailed")
# }
# rbf.regression.test()
#
