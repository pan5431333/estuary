package estuary.components.layers

import breeze.linalg.DenseMatrix
import breeze.numerics.sigmoid

trait SigmoidActivator extends Activator{
  override def activate(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    sigmoid(zCurrent)
  }

  override def activateGrad(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    val sigmoided = sigmoid(zCurrent)
    sigmoided *:* (1.0 - sigmoided)
  }
}
