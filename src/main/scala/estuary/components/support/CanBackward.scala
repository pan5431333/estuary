package estuary.components.support

import estuary.components.regularizer.Regularizer

trait CanBackward[By, Input, Output] {
  def backward(input: Input, by: By, regularizer: Option[Regularizer]): Output
}
