package estuary.components.regularizer

import breeze.linalg.DenseMatrix

/**
  * Created by mengpan on 2017/9/5.
  */
trait Regularizer {
  def getReguCost(m: DenseMatrix[Double]*): Double

  def getReguCostGrad(w: DenseMatrix[Double]): DenseMatrix[Double]

  var lambda: Double = 0.7

  def setLambda(lambda: Double): this.type = {
    if (lambda < 0) throw new IllegalArgumentException("Lambda must be nonnegative!")

    this.lambda = lambda
    this
  }
}
