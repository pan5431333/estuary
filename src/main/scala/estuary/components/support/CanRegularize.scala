package estuary.components.support

import estuary.components.regularizer.Regularizer

trait CanRegularize[-Param] {
  def regu(param: Param, regularizer: Option[Regularizer]): Double
}

