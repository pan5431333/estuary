package estuary.components.layers

import breeze.linalg.DenseMatrix
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.Layer.ForPrediction
import estuary.components.regularizer.Regularizer
import estuary.components.support._


class DropoutLayer(override val numHiddenUnits: Int, val dropoutRate: Double)
  extends Layer with LayerLike[DropoutLayer] with DropoutActivator {

  /** Cache processed data */
  protected[estuary] var yPrevious: DenseMatrix[Double] = _

  def copyStructure: DropoutLayer = new DropoutLayer(numHiddenUnits, dropoutRate)
}

object DropoutLayer {
  def apply(numHiddenUnits: Int, dropoutRate: Double): DropoutLayer = {
    new DropoutLayer(numHiddenUnits, dropoutRate)
  }

  implicit val dropoutLayerCanSetParamNone: CanSetParam[DropoutLayer, None.type] = (_, _) => {}

  implicit val dropoutLayerCanSetParamMatrix: CanSetParam[DropoutLayer, DenseMatrix[Double]] = (_, _) => {}

  implicit val dropoutLayerCanExportParamNone: CanExportParam[DropoutLayer, None.type] = (_) => None

  implicit val dropoutLayerCanExportParamMatrix: CanExportParam[DropoutLayer, Option[DenseMatrix[Double]]] = (_) => None

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

  implicit val dropoutLayerCanRegularize: CanRegularize[DropoutLayer] =
    (_: DropoutLayer, _: Option[Regularizer]) => 0.0

}

