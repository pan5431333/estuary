package estuary.components.regularizer

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.pow

/**
  * Created by mengpan on 2017/9/5.
  */
class L2Regularizer extends Regularizer {

  override def getReguCost(ms: DenseMatrix[Double]*): Double = {
    ms.foldLeft[Double](0.0) { (total, m) =>
      total + sum(pow(m, 2)) / 2.0
    }
  }

  override def getReguCostGrad(w: DenseMatrix[Double]): DenseMatrix[Double] =
    this.lambda * w

}

object L2Regularizer {
  def apply(lambda: Double): L2Regularizer = {
    new L2Regularizer()
      .setLambda(lambda)
  }
}