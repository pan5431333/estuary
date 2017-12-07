package estuary.components.layers

import breeze.linalg.DenseMatrix
import estuary.components.layers.LayerLike.ForPrediction
import estuary.components.regularizer.Regularizer
import estuary.components.support._

/**
  * Created by mengpan on 2017/9/7.
  */
class DropoutLayer(override val numHiddenUnits: Int, val dropoutRate: Double)
  extends Layer[None.type] with LayerLike[None.type, DropoutLayer] with DropoutActivator {

  /** Set parameters received from optimizer */
  override def setParam[O](param: O)(implicit op: CanSetParam[DropoutLayer, O]): Unit = op.set(param, repr)

  override def getReguCost(regularizer: Option[Regularizer])
                          (implicit op: CanRegularize[None.type]): Double = op.regu(None, regularizer)

  /** Cache processed data */
  protected[estuary] var yPrevious: DenseMatrix[Double] = _

  override def hasParams = false

  def copyStructure: DropoutLayer = new DropoutLayer(numHiddenUnits, dropoutRate)
}

object DropoutLayer {
  def apply(numHiddenUnits: Int, dropoutRate: Double): DropoutLayer = {
    new DropoutLayer(numHiddenUnits, dropoutRate)
  }

  implicit val dropoutLayerCanSetParam: CanSetParam[DropoutLayer, None.type] = (_, _) => None

  implicit val dropoutLayerCanExportParam: CanExportParam[DropoutLayer, None.type] = (_) => None

  implicit val dropoutLayerCanAutoInit: CanAutoInit[DropoutLayer] = (_, _) => {}

  implicit val dropoutLayerCanForward: CanForward[DropoutLayer, DenseMatrix[Double], DenseMatrix[Double]] =
    (input, by) => {
      by.yPrevious = input
      by.activate(input)
    }

  implicit val dropoutLayerCanForwardForPrediction: CanForward[DropoutLayer, ForPrediction[DenseMatrix[Double]], DenseMatrix[Double]] =
    (input, _) => input.input

  implicit val dropoutLayerCanBackward: CanBackward[DropoutLayer, DenseMatrix[Double], (DenseMatrix[Double], None.type)] =
    (input, by, regularizer) => {
      val filterMat = by.activateGrad(by.yPrevious)
      (input *:* filterMat, None)
    }
}
