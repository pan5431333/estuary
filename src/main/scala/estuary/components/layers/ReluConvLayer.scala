package estuary.components.layers
import estuary.components.layers.ConvLayer.{ConvSize, Filter}

class ReluConvLayer(override val param: Filter) extends ConvLayer with ReluActivator{

  override def copyStructure: ReluConvLayer = new ReluConvLayer(param).setPreConvSize(preConvSize)
}

object ReluConvLayer {
  def apply(filter: Filter, preConvSize: ConvSize): ReluConvLayer = {
    new ReluConvLayer(filter).setPreConvSize(preConvSize)
  }
}