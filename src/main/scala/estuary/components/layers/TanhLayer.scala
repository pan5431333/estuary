package estuary.components.layers

/**
  * Created by mengpan on 2017/8/26.
  */
class TanhLayer(val numHiddenUnits: Int, val batchNorm: Boolean) extends ClassicLayer with TanhActivator{

  def copyStructure: TanhLayer = new TanhLayer(numHiddenUnits, batchNorm).setPreviousHiddenUnits(previousHiddenUnits)
}

object TanhLayer {
  def apply(numHiddenUnits: Int, batchNorm: Boolean = false): TanhLayer = {
    new TanhLayer(numHiddenUnits, batchNorm)
  }
}

