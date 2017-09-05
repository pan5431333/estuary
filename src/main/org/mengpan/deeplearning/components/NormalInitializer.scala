package org.mengpan.deeplearning.components

/**
  * Created by mengpan on 2017/9/5.
  */
object NormalInitializer extends WeightsInitializer{
  override protected def getWeightsMultipliyer(previousLayerDim: Int, currentLayerDim: Int): Double = 0.01
}
