package estuary.components.layers

trait Layer[+Param] extends LayerLike[Param, Layer[Param]]{
  val numHiddenUnits: Int

  /**Used to distribute model instances onto multiple machines for distributed optimization*/
  def copyStructure: Layer[Param]
}
