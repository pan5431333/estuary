package estuary.components.support

trait CanForward[By, Input, Output] {
  def forward(input: Input, by: By): Output
}
