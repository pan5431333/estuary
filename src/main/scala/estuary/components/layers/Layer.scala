package estuary.components.layers

import breeze.linalg.DenseMatrix
import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.Layer.ForPrediction
import estuary.components.regularizer.Regularizer
import estuary.components.support._


trait LayerLike[+Repr <: Layer] extends Serializable {
  /** Make it more convenient to write code in type class design pattern */
  def repr: Repr = this.asInstanceOf[Repr]

  def setParam[TT >: Repr](param: DenseMatrix[Double])(implicit op: CanSetParam[TT, DenseMatrix[Double]]): Unit =
    op.set(param, repr)

  def getParam[TT >: Repr](implicit op: CanExportParam[TT, Option[DenseMatrix[Double]]]): Option[DenseMatrix[Double]] = op.export(repr)

  def init[TT >: Repr](initializer: WeightsInitializer)(implicit op: CanAutoInit[TT]): Unit =
    op.init(repr, initializer)

  def forward[TT >: Repr, Input, Output](yPrevious: Input)(implicit op: CanForward[TT, Input, Output]): Output =
    op.forward(yPrevious, repr)

  def forwardForPrediction[TT >: Repr, Input, Output](yPrevious: Input)
                                                     (implicit op: CanForward[TT, ForPrediction[Input], Output]): Output =
    op.forward(ForPrediction(yPrevious), repr)

  def backward[TT >: Repr, BackwardInput, BackwardOutput](dYCurrent: BackwardInput, regularizer: Option[Regularizer])
                                                         (implicit op: CanBackward[TT, BackwardInput, BackwardOutput]): BackwardOutput =
    op.backward(dYCurrent, repr, regularizer)

  def getReguCost[TT >: Repr](regularizer: Option[Regularizer])(implicit op: CanRegularize[TT]): Double =
    op.regu(repr, regularizer)
}

object Layer {

  case class ForPrediction[Input](input: Input)

}


trait Layer extends LayerLike[Layer] {
  val numHiddenUnits: Int

  /** Used to distribute model instances onto multiple machines for distributed optimization */
  def copyStructure: Layer
}


