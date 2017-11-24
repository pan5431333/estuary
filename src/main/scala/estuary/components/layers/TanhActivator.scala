package estuary.components.layers

import breeze.linalg.DenseMatrix
import breeze.numerics.{pow, tanh}

trait TanhActivator extends Activator{
  protected def activate(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    tanh(zCurrent)
  }

  protected def activateGrad(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    1.0 - pow(zCurrent, 2)
  }
}
