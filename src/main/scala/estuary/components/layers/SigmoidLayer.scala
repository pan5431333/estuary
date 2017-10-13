package estuary.components.layers

import breeze.linalg.DenseMatrix
import breeze.numerics.sigmoid

/**
  * Created by mengpan on 2017/8/26.
  */
class SigmoidLayer extends Layer {

  override def activationFuncEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    sigmoid(zCurrent)
  }

  override def activationGradEval(zCurrent: DenseMatrix[Double]): DenseMatrix[Double] = {
    val sigmoided = sigmoid(zCurrent)
    sigmoided *:* (1.0 - sigmoided)
  }

  def copyStructure: SigmoidLayer = new SigmoidLayer().setBatchNorm(batchNorm).setNumHiddenUnits(numHiddenUnits).setPreviousHiddenUnits(previousHiddenUnits)
}

object SigmoidLayer {
  def apply(numHiddenUnits: Int, batchNorm: Boolean = false): SigmoidLayer = {
    new SigmoidLayer()
      .setNumHiddenUnits(numHiddenUnits)
      .setBatchNorm(batchNorm)
  }
}

