package org.mengpan.deeplearning.components.initializer

import breeze.numerics.sqrt

/**
  * Created by mengpan on 2017/9/5.
  */
object HeInitializer extends WeightsInitializer{
  override protected def getWeightsMultipliyer(previousLayerDim: Int,
                                               currentLayerDim: Int):
  Double = {
    sqrt(2.0/previousLayerDim)
  }
}
