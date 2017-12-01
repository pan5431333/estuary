package estuary.components.support

trait CanAutoInit[By, Shape, To] {
  def init(shape: Shape, initializer: By): To
}
