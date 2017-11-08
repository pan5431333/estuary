package estuary.components.layers
import estuary.components.layers.ConvLayer.{ConvSize, Filter}

class ReluConvLayer(val filter: Filter) extends ConvLayer with ReluActivator{
  override protected var preConvSize: ConvSize = _

  override def copyStructure: ReluConvLayer = new ReluConvLayer(filter).setPreConvSize(preConvSize)
}

object ReluConvLayer {
  def apply(filter: Filter, preConvSize: ConvSize): ReluConvLayer = {
    new ReluConvLayer(filter).setPreConvSize(preConvSize)
  }
}