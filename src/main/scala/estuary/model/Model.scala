package estuary.model

trait Model[Params] extends ModelLike[Params, Model[Params]] {
  def copyStructure: Model[Params]
}
