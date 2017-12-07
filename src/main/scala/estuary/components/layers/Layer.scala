package estuary.components.layers

trait Layer extends LayerLike[Layer]{
  val numHiddenUnits: Int

  /**Used to distribute model instances onto multiple machines for distributed optimization*/
  def copyStructure: Layer
}
