package org.mengpan.deeplearning.components.initializer

import breeze.numerics.sqrt

/**
  * Created by mengpan on 2017/9/5.
  */
object XaiverInitializer extends WeightsInitializer{
  override protected def getWeightsMultipliyer(previousLayerDim: Int, currentLayerDim: Int): Double = {
    sqrt(1.0/previousLayerDim)
  }
}
