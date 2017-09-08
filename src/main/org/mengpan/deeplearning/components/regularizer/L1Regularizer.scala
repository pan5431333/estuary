package org.mengpan.deeplearning.components.regularizer

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.{abs, pow}

/**
  * Created by mengpan on 2017/9/5.
  */
class L1Regularizer extends Regularizer{

  override def getReguCost(paramsList:
                                     List[(DenseMatrix[Double], DenseVector[Double])]):
  Double = {
    paramsList
        .foldLeft[Double](0.0){(total, params) => total + sum(abs(params._1))}
  }

  override def getReguCostGrad(w: DenseMatrix[Double],
                                         numExamples: Int):
  DenseMatrix[Double] = this.lambda / numExamples.toDouble * sign(w)

  private def sign(w: DenseMatrix[Double]): DenseMatrix[Double] = {
    w.map(e => if (e > 0) 1.0 else if (e < 0) -1.0 else 0.0)
  }
}
