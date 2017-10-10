package estuary.components.regularizer

import breeze.linalg.DenseMatrix

/**
  * Created by mengpan on 2017/9/5.
  */
object VoidRegularizer extends Regularizer {
  override def getReguCost(m: DenseMatrix[Double]*): Double = 0.0

  override def getReguCostGrad(w: DenseMatrix[Double]): DenseMatrix[Double] = DenseMatrix.zeros[Double](w.rows, w.cols)
}
