package estuary.components.layers

import breeze.linalg.DenseMatrix
import breeze.numerics.{pow, tanh}

/**
  * Created by mengpan on 2017/8/26.
  */
class TanhLayer(val numHiddenUnits: Int, val batchNorm: Boolean) extends Layer {
  protected def activationFuncEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    tanh(zCurrent)
  }

  protected def activationGradEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    1.0 - pow(zCurrent, 2)
  }

  def copyStructure: TanhLayer = new TanhLayer(numHiddenUnits, batchNorm).setPreviousHiddenUnits(previousHiddenUnits).asInstanceOf[TanhLayer]

  override def updateNumHiddenUnits(numHiddenUnits: Int) = new TanhLayer(numHiddenUnits, batchNorm)
}

object TanhLayer {
  def apply(numHiddenUnits: Int, batchNorm: Boolean = false): TanhLayer = {
    new TanhLayer(numHiddenUnits, batchNorm)
  }
}

