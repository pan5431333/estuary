package estuary.components.support

trait CanSetParam[For, From, To] {
  def set(from: From, foor: For): To
}
