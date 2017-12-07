package estuary.support

import estuary.components.optimizer.Optimizer

trait CanTrain[-Repr, -Feature, -Label, Param] {
  def train(feature: Feature, label: Label, initParam: Param, optimizer: Optimizer, by: Repr): Param
}
