package estuary.components.layers

import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.LayerLike.ForPrediction
import estuary.components.regularizer.Regularizer
import estuary.components.support._

trait LayerLike[+Repr <: Layer] extends Serializable {
  /**Make it more convenient to write code in type class design pattern */
  def repr: Repr = this.asInstanceOf[Repr]

  /**Set parameters received from optimizer*/
  def setParam[O](param: O)(implicit op: CanSetParam[Repr, O]): Unit = op.set(param, repr)

  def getParam[O](implicit op: CanExportParam[Repr, O]): O = op.export(repr)

  def init(initializer: WeightsInitializer)(implicit op: CanAutoInit[Repr]): Unit = op.init(repr, initializer)

  def forward[Input, Output](yPrevious: Input)(implicit op: CanForward[Repr, Input, Output]): Output =
    op.forward(yPrevious, repr)

  def forwardForPrediction[Input, Output](yPrevious: Input)
                                         (implicit op: CanForward[Repr, ForPrediction[Input], Output]): Output =
    op.forward(ForPrediction(yPrevious), repr)

  def backward[BackwardInput, BackwardOutput](dYCurrent: BackwardInput, regularizer: Option[Regularizer])
                                             (implicit op: CanBackward[Repr, BackwardInput, BackwardOutput]): BackwardOutput =
    op.backward(dYCurrent, repr, regularizer)

  def getReguCost(regularizer: Option[Regularizer])(implicit op: CanRegularize[Repr]): Double = op.regu(repr, regularizer)
}

object LayerLike {
  case class ForPrediction[Input](input: Input)
}
