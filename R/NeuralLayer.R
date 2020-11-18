# "R" implementation of Neural Network layer, done in OOP fashion.
#
# Kary Främling, created 28 dec 2005
#

#-----
# "Activation" functions here, i.e. the one that determines activation level
# as a function of input values.
#-----

#' Weighted sum activation function.
#'
#' The most common activation function for neurons.
#'
#' @param input blabla
#' @param weights blabla
#'
#' @return Weighted sum of input, weights.
#' @export
weighted.sum.activation <- function(input, weights) {
  res <- crossprod(t(input),t(weights))
  res
}

#' Squared Euclidean distance between weights and inputs.
#'
#' @param input blabla
#' @param weights blabla
#'
#' @return Squared Euclidean distance
#' @export
squared.distance.activation <- function(input, weights) {
  # Loops are bad but I can't find better solutions just now...
  d2 <- matrix(0,nrow=nrow(input),ncol=nrow(weights))
  for ( i in 1:nrow(input) ) {
    diff <- weights - matrix(input[i,], nrow=nrow(weights), ncol=ncol(weights), byrow=T)
#    diff <- weights - input[i,]
    d2[i,] <- rowSums(diff*diff)
  }
  d2
}

#-----
# "Output" functions here, i.e. functions that take activation value
# and transform it into output value.
#-----

#' Output is same as activation.
#'
#' For neurons who only do a weighted sum.
#'
#' @param activation blabla
#'
#' @return "activation" value as such.
#' @export
identity.output.function <- function(activation) {
  activation
}

#' Sigmoid output function.
#'
#' @param activation blabla
#'
#' @return Sigmoid output value.
#' @export
sigmoid.output.function <- function(activation) {
  res <- 1/(1+exp(-activation))
  res
}

#' Gaussian output function.
#'
#' Spread parameter may be a single value or a vector, i.e.
#' same spread for all or individual spread parameters.
#' The activation value should normally be the squared distance.
#'
#' @param activation blabla
#' @param spread blabla
#'
#' @return Gaussian output value.
#' @export
gaussian.output.function <- function(activation, spread=1.0) {
  res <- exp(-activation/spread)
  res
}

#' Inverse MultiQuadric Equations function.
#'
#' Spread parameter may be a single value or a vector, i.e. same spread
#' for all or individual spread parameters. The activation value should
#' normally be the squared distance.
#'
#' @param activation blabla
#' @param spread blabla
#'
#' @return IMQE value.
#' @export
imqe.output.function <- function(activation, spread=1.0) {
  res <- 1/sqrt(1 + activation/spread)
  res
}

#-----
# Then we also need "inverse" functions for gradient descent.
#-----

#' Neural layer object implementation.
#'
#' @param nbrInputs blabla
#' @param nbrOutputs blabla
#' @param activation.function blabla
#' @param output.function blabla
#' @param use.trace blabla
#' @param use.bias blabla
#'
#' @return Object of class NeuralLayer (which also inherits FunctionApproximator)
#' @export
#' @examples
#' # Create simple weighted sum layer (one neuron) with fixed neurons and
#' # call "eval" for input values (1,1).
#' l <- neural.layer.new(2, 1, weighted.sum.activation, identity.output.function)
#' l$set.weights(matrix(c(1, 2), nrow=1, ncol=2))
#' l$eval(c(1,1))
#'
#' # Plot what it looks like (needs "graphics")
#' x <- y <-seq(0,1,0.05)
#' m <- expand.grid(x,y)
#' persp(x, y, matrix(l$eval(m), nrow=length(x)), ticktype="detailed")
#'
#' # Test squared distance. One input, two "kernels".
#' l <- neural.layer.new(1, 2, squared.distance.activation, imqe.output.function)
#' l$set.weights(matrix(c(1, 2), nrow=2, ncol=1, byrow=TRUE))
#' l$set.spread(1.0)
#' l$eval(c(2))
#' x <- matrix(seq(0,3,0.1), ncol=1)
#' plot(x, l$eval(x)[,1], type="l")
#' lines(x, l$eval(x)[,2], col="blue")
#'
#' # Study output function shapes a little
#' # IMQE versus Gaussian
#' s<-seq(0,100,0.1)
#' plot(s,imqe.output.function(s),type='l', col='black', ylim=c(0,1))
#' lines(s,gaussian.output.function(s),col='green')
#'
#' # Sigmoid
#' s<-seq(-5,5,0.1)
#' plot(s,sigmoid.output.function(s),type='l', col='black', ylim=c(0,1))
neural.layer.new <- function(nbrInputs, nbrOutputs, activation.function, output.function, use.trace=FALSE, use.bias=FALSE) {

  weights <- matrix(0, nrow = nbrOutputs, ncol = nbrInputs)
  afunction <- activation.function
  ofunction <- output.function
  activations <- c()
  outputs <- c()
  inputs <- c()
  targets <- c()
  lr <- 0.1 # Learning rate
  spread <- 1.0 # For Kernel output functions
  normalize <- FALSE # No normalisation of output values by default

  # "Eligibility trace" for reinforcement learning purposes.
  if ( use.trace )
    trace <- eligibility.trace.new(nbrInputs, nbrOutputs)
  else
    trace <- NULL

  # Evaluate output values for the given input values. The input
  # values are organised in columns, one row per sample.
  eval <- function(invals) {
    if ( is.vector(invals) )
      inputs <<- matrix(invals, nrow=1)
    else
      inputs <<- as.matrix(invals)

    if ( use.bias ) { # Add one more column to inputs. Current solution seems clumsy but will have to do for the moment...
      dinp <- dim(inputs)
      bias.matrix <- matrix(1, dinp[1], dinp[2]+1)
      bias.matrix[1:dinp[1],1:dinp[2]] <- inputs
      inputs <<- bias.matrix
    }
    activations <<- afunction(inputs, weights)
    if ( sum(names(formals(ofunction)) == "spread") > 0 )
      outputs <<- ofunction(activations, spread)
    else
      outputs <<- ofunction(activations)
    if ( normalize )
       outputs <<- normalize.to.one(outputs)
    return(outputs)
  }

  # Return list of "public" methods. Also set "class" attribute.
  pub <- list(
              eval = function(invals) { eval(invals) },
              get.inputs = function() { inputs },
              set.inputs = function(inps) { inputs <<- inps },
              get.outputs = function() { outputs },
              set.outputs = function(outps) { outputs <<- outps },
              get.targets = function() { targets },
              get.weights = function() { weights },
              get.activation.function = function() { afunction },
              get.output.function = function() { ofunction },
              get.trace = function() { trace },
              get.lr = function() { lr },
              get.spread = function() { spread },
              get.normalize = function() { normalize },
              get.use.bias = function() { use.bias },
              set.weights = function(w) { weights <<- w },
              set.trace = function(tr) { trace <<- tr },
              set.lr = function(lrate) { lr <<- lrate },
              set.spread = function(s) { spread <<- s },
              set.normalize = function(value) { normalize <<- value },
              set.use.bias = function(value) { use.bias <<- value }
              )

  # We also implement "FunctionApproximator"
  fa <- function.approximator.new()

  class(pub) <- c("NeuralLayer",class(fa))
  return(pub)
}

