package estuary.components.layers

/**
  * Created by mengpan on 2017/8/26.
  */
class SigmoidLayer(val numHiddenUnits: Int, val batchNorm: Boolean) extends ClassicLayer with SigmoidActivator{

  def copyStructure: SigmoidLayer = new SigmoidLayer(numHiddenUnits, batchNorm).setPreviousHiddenUnits(previousHiddenUnits)
}

object SigmoidLayer {
  def apply(numHiddenUnits: Int, batchNorm: Boolean = false): SigmoidLayer = {
    new SigmoidLayer(numHiddenUnits, batchNorm)
  }
}

