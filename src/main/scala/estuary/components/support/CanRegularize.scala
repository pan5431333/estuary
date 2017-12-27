package estuary.components.support

import estuary.components.regularizer.Regularizer

trait CanRegularize[Repr] {
  def regu(foor: Repr, regularizer: Option[Regularizer]): Double
}

