package estuary.components.layers

import breeze.linalg.DenseMatrix

trait Activator {
  protected def activate(zCurrent: DenseMatrix[Double]): DenseMatrix[Double]
  protected def activateGrad(zCurrent: DenseMatrix[Double]): DenseMatrix[Double]
}
