#' Re-create `RBF` object from `inka` object.
#'
#' @param inka blabla
#' @param activation.function blabla
#' @param output.function blabla
#'
#' @return `RBF` object
#' @export
#'
#' @seealso [rbf.new]
#' @examples
#' rbf <- train.inka.formula(Species~., data=iris, spread=0.1, max.iter=20,
#' classification.error.limit=0)
#' predict(rbf, newdata=iris[100,])
#' inka <- rbf$as.inka()
#' newrbf <- rbf.restore(inka)
#' predict(newrbf, newdata=iris[100,])
rbf.restore <- function(inka, activation.function=squared.distance.activation,
                         output.function=imqe.output.function) {
  # Check that we have valid parameter.
  stopifnot(inherits(inka, "inka"))

  nbr.inputs <- ncol(inka$weights[[1]])
  nbr.outputs <- nrow(inka$weights[[1]])
  rbf <- rbf.new(nbr.inputs, nbr.outputs, 0,
                 activation.function = activation.function,
                 output.function = output.function,
                 normalize = inka$normalize,
                 spread = inka$spread)
  hl <- rbf$get.hidden()
  hl$set.weights(inka$weights[[1]])
  ol <- rbf$get.outlayer()
  ol$set.weights(inka$weights[[2]])
  rbf$set.formula(inka$formula)
  return(rbf)
}
