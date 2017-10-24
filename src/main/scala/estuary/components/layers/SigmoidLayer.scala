package estuary.components.layers

import breeze.linalg.DenseMatrix
import breeze.numerics.sigmoid

/**
  * Created by mengpan on 2017/8/26.
  */
class SigmoidLayer(val numHiddenUnits: Int, val batchNorm: Boolean) extends Layer {

  override def activationFuncEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    sigmoid(zCurrent)
  }

  override def activationGradEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    val sigmoided = sigmoid(zCurrent)
    sigmoided *:* (1.0 - sigmoided)
  }

  def copyStructure: SigmoidLayer = new SigmoidLayer(numHiddenUnits, batchNorm).setPreviousHiddenUnits(previousHiddenUnits).asInstanceOf[SigmoidLayer]

  override def updateNumHiddenUnits(numHiddenUnits: Int) = new SigmoidLayer(numHiddenUnits, batchNorm)
}

object SigmoidLayer {
  def apply(numHiddenUnits: Int, batchNorm: Boolean = false): SigmoidLayer = {
    new SigmoidLayer(numHiddenUnits, batchNorm)
  }
}

