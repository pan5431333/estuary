package estuary.components.regularizer

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.abs

/**
  * Created by mengpan on 2017/9/5.
  */
class L1Regularizer extends Regularizer {

  override def getReguCost(m: DenseMatrix[Double]*): Double = {
    m.foldLeft[Double](0.0) { (total, e) => total + sum(abs(e)) }
  }

  override def getReguCostGrad(w: DenseMatrix[Double]): DenseMatrix[Double] = this.lambda * sign(w)

  private def sign(w: DenseMatrix[Double]): DenseMatrix[Double] = {
    val res = DenseMatrix.zeros[Double](w.rows, w.cols)
    for (i <- (0 until w.rows).par) {
      for (j <- (0 until w.cols).par) {
        res(i, j) = if (w(i, j) > 0) 1.0 else if (w(i, j) < 0) -1.0 else 0.0
      }
    }
    res
  }
}

object L1Regularizer {
  def apply(lambda: Double): L1Regularizer = {
    new L1Regularizer()
      .setLambda(lambda)
  }
}
