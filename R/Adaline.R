# "R" implementation of Adaline, done in OOP fashion. This is
# implemented as a "sub-class" of NeuralLayer, so the only thing to
# add here is training methods.
#
# Kary Främling, created September 2005
#
#' Create Adaline neural network
#'
#' @param nbrInputs Number of inputs.
#' @param nbrOutputs Number of outputs
#' @param use.trace Use eligibility trace or not.
#' Only for reinforcement learning.
#'
#' @return Object of class Adaline (and inherited classes)
#' @export
#' @examples
#' # Create Adaline with two inputs, 3 outputs, assign weights manually.
#' a <- adaline.new(2, 3)
#' weights <- matrix(c(0, 0, 1, 1, 2, 2), nrow=3, ncol=2, byrow = TRUE)
#' a$set.weights(weights)
#' inputs <- c(1, 1.5)
#' a$eval(inputs)
#'
#' # Plot what it looks like for output 2.
#' x <- y <-seq(0,1,0.05)
#' m <- expand.grid(x,y)
#' persp(x, y, matrix(a$eval(m)[,2], nrow=length(x)), ticktype="detailed")
#'
#' # Small training test
#' inputs <- c(1,1.5)
#' targets <- c(1, 2, 3)
#' a$eval(inputs) # Have to call eval() before calling train()
#' a$train(targets)
#' a$eval(inputs) # Should have changed a little
#' persp(x, y, matrix(a$eval(m)[,2], nrow=length(x)), ticktype="detailed")
#'
#' # Slightly more training
#' inputs <- matrix(c(0,0,1,1), nrow=2, byrow=TRUE)
#' a$eval(inputs)
#' targets <- matrix(c(0,0,0,1,2,3), nrow=2, byrow=TRUE)
#' for ( i in 1:100 ) {
#'   a$eval(inputs[1,])
#'   a$train(targets[1,])
#'   a$eval(inputs[2,])
#'   a$train(targets[2,])
#' }
#' a$eval(inputs)
#' persp(x, y, matrix(a$eval(m)[,2], nrow=length(x)), ticktype="detailed")
adaline.new <- function(nbrInputs, nbrOutputs, use.trace=FALSE) {

  # Set up "super" class and get its environment. We need the environment
  # only if we need to access some variables directly, not only through
  # the "public" interface.
  # "super" has to have at least one "method" for getting the environment.
  super <- neural.layer.new(nbrInputs, nbrOutputs, weighted.sum.activation, identity.output.function, use.trace)
  se <- environment(super[[1]])

  # Set up our "own" instance variables.
  targets <- NULL
  nlms <- F # Use standard LMS by default
  mixtw <- 1.0 # Weight of this Adaline when used in some
               #kind of "mixture model", e.g. BIMM

  # Perform LMS or NLMS training. No return value.
  # CAN BE APPLIED TO ONLY ONE SAMPLE AT A TIME! NOT TO A WHOLE SET!!
  train <- function(t) {
    targets <<- t
    inps <- super$get.inputs()
    outps <- super$get.outputs()

    # Sanity check
    if ( is.null(inps) || is.null(outps) )
      stop("input or output values are NULL. Did you do forward pass (eval()) before calling train?")

    lr <- super$get.lr()
    # Widrow-Hoff here
    if ( !nlms ) {
      delta <- as.vector(lr*(targets - outps))%o%as.vector(inps)
    }
    else {
      nfact <- matrix(mixtw*(inps%*%t(inps)),
                      nrow=nrow(inps), ncol=ncol(inps))
      delta <- as.vector(lr*(targets - outps))%o%as.vector(inps/nfact)
    }
    super$set.weights(super$get.weights() + delta)
  }

  # Perform LMS or NLMS training for given "delta" value that indicates
  # the error. "delta" can be a constant.
  # Should be modified so that it can also be a vector. Having a vector
  # is useful for "ordinary" learning where the error value is usually
  # different for every output.
  # In Q-learning, the error value is global for the whole net, i.e. for
  # all outputs. Then the extent to which this error is distributed to
  # different outputs depends on the "eligibility trace". The
  # the expression for "delta" is:
  # r(t+1) + gamma*Q(s(t+1),a(t+1)) - Q(s(t),a(t))
  # IMPORTANT!?? This function can only be used for discrete
  # state spaces with lookup-table type of calculations. This means
  # that sum(inputs^2) = 1, so NLMS only signifies using "K" factor.
  train.with.delta <- function(delta) {
    if ( is.null(trace) ) {
      tr <- 1.0
    }
    else {
      tr <- trace$get.trace()
    }
    d <- super$get.lr()*delta*tr
    if ( nlms ) {
      nfact <- matrix(mixtw*(super$get.inputs()%*%t(super$get.inputs())), nrow=nrow(d), ncol=ncol(d))
      d <- d/nfact
    }

    super$set.weights(super$get.weights() + d)
  }

  # Construct list of "public methods"
  pub <- list(
              get.nlms = function() { nlms },
              get.mixtw = function() { mixtw },
              set.nlms = function(value) { nlms <<- value },
              set.mixtw = function(value) { mixtw <<- value },
              train.with.delta = function(delta) {
                train.with.delta(delta)
              },
              train = function(t) { train(t) }
              )

  # Set up the environment so that "private" variables in "super" become
  # visible. This might not always be a good choice but it is the most
  # convenient here for the moment.
  parent.env(environment(pub[[1]])) <- se

  # We return the list of "public" methods. Since we "inherit" from
  # "super" (NeuralLayer), we need to concatenate our list of "methods"
  # with the inherited ones.
  # Also add something to our "class". Might be useful in the future
  methods <- c(pub,super)

  # Also declare that we implement the "TrainableApproximator" interface
  class(methods) <- c("Adaline",class(super),class(trainable.approximator.new()))

  return(methods)
}

