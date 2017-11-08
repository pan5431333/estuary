package estuary.components.layers

import breeze.linalg.DenseMatrix
import breeze.numerics.sigmoid

trait SigmoidActivator extends Activator{
  override def activationFuncEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    sigmoid(zCurrent)
  }

  override def activationGradEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    val sigmoided = sigmoid(zCurrent)
    sigmoided *:* (1.0 - sigmoided)
  }
}
