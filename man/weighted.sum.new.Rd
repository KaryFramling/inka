% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Functions.R
\name{weighted.sum.new}
\alias{weighted.sum.new}
\title{"Objectified" Weighted sum
Weighted sum objectified". Weights must be given as one-column matrix.
The number of columns in input matrix must be same
as number of rows of w.}
\usage{
weighted.sum.new(weights)
}
\arguments{
\item{weights}{blabla}
}
\value{
Object of class FunctionApproximator
}
\description{
"Objectified" Weighted sum
Weighted sum objectified". Weights must be given as one-column matrix.
The number of columns in input matrix must be same
as number of rows of w.
}
\examples{
ws<-weighted.sum.new(matrix(c(0.2,0.8),ncol=1))
ws$eval(matrix(c(0.5,0.5,1,1), ncol=2, byrow=TRUE))
}
