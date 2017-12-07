package estuary.support

import estuary.components.optimizer.Optimizer

trait CanTrain[-Repr, -Feature, -Label] {
  def train(feature: Feature, label: Label, optimizer: Optimizer, by: Repr): Unit
}
