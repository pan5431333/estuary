package estuary.components.layers

import breeze.linalg.DenseMatrix
import breeze.numerics.{pow, tanh}

trait TanhActivator extends Activator{
  protected def activationFuncEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    tanh(zCurrent)
  }

  protected def activationGradEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    1.0 - pow(zCurrent, 2)
  }
}
