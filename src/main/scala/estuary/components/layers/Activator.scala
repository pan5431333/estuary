package estuary.components.layers

import breeze.linalg.DenseMatrix

trait Activator {
  def activate(zCurrent: DenseMatrix[Double]): DenseMatrix[Double]
  def activateGrad(zCurrent: DenseMatrix[Double]): DenseMatrix[Double]
}
