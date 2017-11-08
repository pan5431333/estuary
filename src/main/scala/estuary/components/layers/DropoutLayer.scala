package estuary.components.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import estuary.components.regularizer.Regularizer

/**
  * Created by mengpan on 2017/9/7.
  */
class DropoutLayer(val numHiddenUnits: Int, val batchNorm: Boolean, val dropoutRate: Double) extends ClassicLayer with DropoutActivator{

  override def forward(yPrevious: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.yPrevious = yPrevious
    activationFuncEval(yPrevious)
  }

  override def backward(dYCurrent: DenseMatrix[Double], regularizer: Option[Regularizer]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val filterMat = activationGradEval(yPrevious)

    (dYCurrent *:* filterMat, DenseMatrix.zeros[Double](previousHiddenUnits + 1, numHiddenUnits))
  }

  def copyStructure: DropoutLayer = {
    new DropoutLayer(numHiddenUnits, batchNorm, dropoutRate).setPreviousHiddenUnits(previousHiddenUnits)
  }
}

object DropoutLayer {
  def apply(numHiddenUnits: Int, dropoutRate: Double): DropoutLayer = {
    new DropoutLayer(numHiddenUnits, false, dropoutRate)
  }
}
