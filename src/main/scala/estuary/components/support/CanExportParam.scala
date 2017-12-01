package estuary.components.support

trait CanExportParam[From, To] {
  def export(from: From): To
}

