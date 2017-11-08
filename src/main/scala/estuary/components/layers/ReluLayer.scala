package estuary.components.layers

/**
  * Created by mengpan on 2017/8/26.
  */
class ReluLayer(val numHiddenUnits: Int, val batchNorm: Boolean) extends ClassicLayer with ReluActivator{

  def copyStructure: ReluLayer = new ReluLayer(numHiddenUnits, batchNorm).setPreviousHiddenUnits(previousHiddenUnits)
}

object ReluLayer {
  def apply(numHiddenUnits: Int, batchNorm: Boolean = false): ReluLayer = {
    new ReluLayer(numHiddenUnits, batchNorm)
  }
}

