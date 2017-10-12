package estuary.components.layers

import breeze.linalg.DenseMatrix
import breeze.numerics.{pow, tanh}

/**
  * Created by mengpan on 2017/8/26.
  */
class TanhLayer extends Layer {

  protected override def activationFuncEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    tanh(zCurrent)
  }

  protected override def activationGradEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    1.0 - pow(zCurrent, 2)
  }

  override def copy = new TanhLayer().setBatchNorm(batchNorm).setNumHiddenUnits(numHiddenUnits).setPreviousHiddenUnits(previousHiddenUnits)
}

object TanhLayer {
  def apply(numHiddenUnits: Int, batchNorm: Boolean = true): TanhLayer = {
    new TanhLayer()
      .setNumHiddenUnits(numHiddenUnits)
      .setBatchNorm(batchNorm)
  }
}

