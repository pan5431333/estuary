package estuary.components.layers

import breeze.linalg.DenseMatrix
import estuary.components.regularizer.Regularizer
import estuary.components.support.{CanBackward, CanForward}

/**
  * Created by mengpan on 2017/9/7.
  */
class DropoutLayer(val numHiddenUnits: Int, val batchNorm: Boolean, val dropoutRate: Double) extends ClassicLayer with DropoutActivator{

  override def forward(yPrevious: DenseMatrix[Double])(implicit op: CanForward[ClassicLayer, DenseMatrix[Double], DenseMatrix[Double]]): DenseMatrix[Double] =
    implicitly[CanForward[DropoutLayer, DenseMatrix[Double], DenseMatrix[Double]]].forward(yPrevious, this)

  override def backward(dYCurrent: DenseMatrix[Double], regularizer: Option[Regularizer])(implicit op: CanBackward[ClassicLayer, DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val filterMat = activateGrad(yPrevious)

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

  implicit val dropoutLayerCanForward: CanForward[DropoutLayer, DenseMatrix[Double], DenseMatrix[Double]] =
    (input, by) => {
      by.yPrevious = input
      by.yPrevious = input
      by.y = by.activate(input)
      by.y
    }
}
