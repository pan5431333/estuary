package estuary.model

import estuary.components.initializer.WeightsInitializer
import estuary.components.layers.LayerLike.ForPrediction
import estuary.components.optimizer.Optimizer
import estuary.components.regularizer.Regularizer
import estuary.components.support._
import estuary.support.CanTrain
import scala.language.higherKinds


trait ModelLike[Repr <: Model] extends Serializable {

  def repr: Repr = this.asInstanceOf[Repr]

  def init(initializer: WeightsInitializer)
          (implicit op: CanAutoInit[Repr]): Unit = op.init(repr, initializer)

  def setParams[Params](params: Params)(implicit op: CanSetParam[Repr, Params]): Unit = op.set(params, repr)

  def getParams[Params](implicit op: CanExportParam[Repr, Params]): Params = op.export(repr)

  def forward[Input, Output](feature: Input)
                            (implicit op: CanForward[Repr, Input, Output]): Output = op.forward(feature, repr)

  def backward[Input, Output](label: Input, regularizer: Option[Regularizer])
                             (implicit op: CanBackward[Repr, Input, Output]): Output = op.backward(label, repr, regularizer)

  def predict[Input, Output](feature: Input)
                            (implicit op: CanForward[Repr, ForPrediction[Input], Output]): Output = op.forward(ForPrediction(feature), repr)

  def train[Feature, Label](feature: Feature, label: Label, optimizer: Optimizer)
                           (implicit op: CanTrain[Repr, Feature, Label]): Unit = op.train(feature, label, optimizer, repr)
}

