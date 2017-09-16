package org.mengpan.deeplearning.components.regularizer

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.{abs, pow}

/**
  * Created by mengpan on 2017/9/5.
  */
class L1Regularizer extends Regularizer{

  override def getReguCost(m: DenseMatrix[Double]*): Double = {
    m.foldLeft[Double](0.0){(total, e) => total + sum(abs(e))}
  }

  override def getReguCostGrad(w: DenseMatrix[Double]): DenseMatrix[Double] = this.lambda * sign(w)

  private def sign(w: DenseMatrix[Double]): DenseMatrix[Double] = {
    w.map(e => if (e > 0) 1.0 else if (e < 0) -1.0 else 0.0)
  }
}

object L1Regularizer {
  def apply(lambda: Double): L1Regularizer = {
    new L1Regularizer()
      .setLambda(lambda)
  }
}
