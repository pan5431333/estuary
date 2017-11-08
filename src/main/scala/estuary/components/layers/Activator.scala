package estuary.components.layers

import breeze.linalg.DenseMatrix

trait Activator {
  protected def activationFuncEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double]
  protected def activationGradEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double]
}
